/*****************************************************************************\
 *  lpsched.c - linear programming based scheduler plugin.
 *
 *  Developed by Seren Soner and Can Ozturan 
 *  from Bogazici University, Istanbul, Turkey
 *  as a part of PRACE WP9.2
 *  Contact seren.soner@gmail.com, ozturaca@boun.edu.tr if necessary
 
 *  Used backfill plugin as base code. There are some functions that are
    left untouched.

 *  For the theory behind the plugin, see the manuscript
 *  doi: ....
\*****************************************************************************/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "lp_lib.h"

#include "slurm/slurm.h"
#include "slurm/slurm_errno.h"

#include "src/common/list.h"
#include "src/common/macros.h"
#include "src/common/node_select.h"
#include "src/common/parse_time.h"
#include "src/common/slurm_protocol_api.h"
#include "src/common/xmalloc.h"
#include "src/common/xstring.h"

#include "src/slurmctld/acct_policy.h"
#include "src/slurmctld/front_end.h"
#include "src/slurmctld/job_scheduler.h"
#include "src/slurmctld/licenses.h"
#include "src/slurmctld/locks.h"
#include "src/slurmctld/node_scheduler.h"
#include "src/slurmctld/preempt.h"
#include "src/slurmctld/reservation.h"
#include "src/slurmctld/slurmctld.h"
#include "src/slurmctld/srun_comm.h"
#include "src/plugins/select/lpconsres/select_lpconsres.h"
#include "lpsched.h"

/* run every 3 seconds */
#ifndef SCHED_INTERVAL
#  define SCHED_INTERVAL	3
#endif

/* max # of jobs = 50 */
#ifndef MAX_JOB_COUNT
#define   MAX_JOB_COUNT 50
#endif

#define SLURMCTLD_THREAD_LIMIT	5

extern int gres_job_gpu_count(List job_gres_list);
/*********************** local variables *********************/
static bool stop_lpsched = false;
static pthread_mutex_t term_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  term_cond = PTHREAD_COND_INITIALIZER;
static bool config_flag = false;
static uint32_t debug_flags = 0;
static int sched_interval = SCHED_INTERVAL;
static int max_job_count = MAX_JOB_COUNT;

/*********************** local functions *********************/
static int  _run_solver_opt(void);
static void _load_config(void);
static void _my_sleep(int secs);
static int  _start_job(struct job_record *job_ptr, bitstr_t *bitmap);
extern struct node_schedinfo *_return_nodes();
extern int print_nodes_lpconsres(struct node_record *node_ptr, int node_cnt);

/* Terminate lpsched_agent */
extern void stop_lpsched_agent(void)
{
	pthread_mutex_lock(&term_lock);
	stop_lpsched = true;
	pthread_cond_signal(&term_cond);
	pthread_mutex_unlock(&term_lock);
}

static void _my_sleep(int secs)
{
	struct timespec ts = {0, 0};

	ts.tv_sec = time(NULL) + secs;
	pthread_mutex_lock(&term_lock);
	if (!stop_lpsched)
		pthread_cond_timedwait(&term_cond, &term_lock, &ts);
	pthread_mutex_unlock(&term_lock);
}

static void _load_config(void)
{
	char *sched_params, *tmp_ptr;

	sched_params = slurm_get_sched_params();
	debug_flags  = slurm_get_debug_flags();

	if (sched_params && (tmp_ptr=strstr(sched_params, "interval=")))
		sched_interval = atoi(tmp_ptr + 9);
	if (sched_interval < 1) {
		fatal("Invalid scheduler interval: %d",
		      sched_interval);
	}

	if (sched_params && (tmp_ptr=strstr(sched_params, "max_job_count=")))
		max_job_count = atoi(tmp_ptr + 11);
	if (max_job_count < 1) {
		fatal("Invalid lpsched max_job_count: %d",
		      max_job_count);
	}
	xfree(sched_params);
}

/* Note that slurm.conf has changed */
extern void lpsched_reconfig(void)
{
	config_flag = true;
}

extern void *lpsched_agent(void *args)
{
	DEF_TIMERS;
	time_t now;
	double wait_time;
	static time_t last_lpsched_time = 0;
	/* Read config and partitions; Write jobs and nodes */
	slurmctld_lock_t all_locks = {
		READ_LOCK, WRITE_LOCK, WRITE_LOCK, READ_LOCK };

	_load_config();
	last_lpsched_time = time(NULL);
	while (!stop_lpsched) {
		_my_sleep(sched_interval);
		if (stop_lpsched)
			break;
		if (config_flag) {
			config_flag = false;
			_load_config();
		}
		now = time(NULL);
		wait_time = difftime(now, last_lpsched_time);
		if (!avail_front_end() || (wait_time < sched_interval))
			continue;
		START_TIMER;
		lock_slurmctld(all_locks);
		while (_run_solver_opt()) ;
		last_lpsched_time = time(NULL);
		unlock_slurmctld(all_locks);
		END_TIMER;
	}
	return NULL;
}

/* Return non-zero to break the lpsched loop if change in job, node or
 * partition state or the lpsched scheduler needs to be stopped. */
static int _yield_locks(void)
{
	slurmctld_lock_t all_locks = {
		READ_LOCK, WRITE_LOCK, WRITE_LOCK, READ_LOCK };
	time_t job_update, node_update, part_update;

	job_update  = last_job_update;
	node_update = last_node_update;
	part_update = last_part_update;

	unlock_slurmctld(all_locks);
	_my_sleep(sched_interval);
	lock_slurmctld(all_locks);

	if ((last_job_update  == job_update)  &&
	    (last_node_update == node_update) &&
	    (last_part_update == part_update) &&
	    (! stop_lpsched) && (! config_flag))
		return 0;
	else
		return 1;
}

int solve_allocation(int nodeSize, int windowSize, int timeout,
		sched_nodeinfo_t *node_array, solver_job_list_t *job_array)
{
	lprec *lp;
	solver_job_list_t *solver_job_ptr;
	int Ncol = (2 * nodeSize + 2) * windowSize;
	int i, j, k;
	int *colno = (int *)malloc(2 * windowSize * sizeof(*colno));
	REAL *row = (REAL *)malloc(2 * windowSize * sizeof(*row));
	REAL *var = (REAL *)malloc(Ncol * sizeof(*var));
	REAL *sparserow;
	int *sparsecol;
	char varname[20];
	/*
	m = nodesize
	n = windowsize
	s_(n) binary variable, selected or not : 1 - n
	x_(m*n) on_node cpu assignment variables : n + 1 : n*(m+1)
	t_(n*m) whether a job uses a node or not : n*(m+1)+1 : n*(2m+1)
	c_(n) cost variable : (n*2m+1) + 1 : n*(2m+2)
	sum: n*(2m+2)
	*/
	lp = make_lp(0,Ncol);
	set_timeout(lp, timeout);
	if (lp == NULL)
		return -1;

	for (j = 0; j < windowSize; j++) {
		sprintf(varname, "s_%d",j+1);
		set_col_name(lp, j + 1, varname);
	}
	for (i = 0; i < nodeSize; i++) {
		for (j = 0; j < windowSize; j++) {		
			sprintf(varname, "x_%d_%d",i+1,j+1);
			set_col_name(lp, (1 + i) * windowSize + 1 + j, varname);
		}
	}
	for (j = 0; j < windowSize; j++) {		
		sprintf(varname, "c_%d",j+1);
		set_col_name(lp, windowSize * (2 * nodeSize + 1) + j + 1, varname);
		for (i = 0; i < nodeSize; i++) {
			sprintf(varname, "t_%d_%d",j+1,i+1);
			set_col_name(lp, windowSize * (nodeSize + 1) + j * nodeSize + 1 + i, varname);
		}
	}

	set_add_rowmode(lp, TRUE);
        /*
	sum over nodes (allocated cpu) should be equal to job's requested cpu
	sum_i(x_ij) == r_j * s_j
	*/
	sparserow = (REAL*)malloc((nodeSize + 1) * sizeof(*sparserow));
	sparsecol = (int*)malloc((nodeSize + 1) * sizeof(*colno));
	for (j = 0; j < windowSize; j++) {
		for (i = 0; i < nodeSize; i++) {
			sparserow[i] = 1.0;
			sparsecol[i] = (1 + i) * windowSize + j + 1;
		}
		sparserow[nodeSize] = (int)(-job_array[j].min_cpus);
		sparsecol[nodeSize] = j + 1;
		add_constraintex(lp, nodeSize + 1, sparserow, sparsecol, EQ, 0);
	}
	free(sparserow);
	free(sparsecol);

        /*
	sum over jobs for cpu should be available 
	sum_j(x_ij) <= R_i
	*/
	sparserow = (REAL*)malloc(windowSize * sizeof(*sparserow));
	sparsecol = (int*)malloc(windowSize * sizeof(*colno));
	for (i = 0; i < nodeSize; i++) {
		for (j = 0; j < windowSize; j++) {
			sparserow[j] = 1.0;
			sparsecol[j] = (1 + i) * windowSize + j + 1;
		}
		add_constraintex(lp, windowSize, sparserow, sparsecol, LE, (int)(node_array[i].rem_cpus));
	}
	free(sparserow);
	free(sparsecol);

        /*
	sum over jobs for gpu should be available on that node
	sum_j(t_ji * g_j) <= G_i
	*/
	sparserow = (REAL*)malloc(windowSize * sizeof(*sparserow));
	sparsecol = (int*)malloc(windowSize * sizeof(*colno));
	for (i = 0; i < nodeSize; i++) {
		for (j = 0; j < windowSize; j++) {
			sparserow[j] = job_array[j].gpu;
			sparsecol[j] = windowSize * (nodeSize + 1) + 1 + j * nodeSize + i;
		}
		add_constraintex(lp, windowSize, sparserow, sparsecol, LE, (int)(node_array[i].rem_gpus));
	}
	free(sparserow);
	free(sparsecol);

	/* 
	FUTURE WORK FOR TOPOLOGY AWARENESS
	t_ji is 1 if job j is allocated resourced in node i
	t_ji = 1 if x_ij > 0
	*/
	sparserow = (REAL*)malloc(2 * sizeof(*sparserow));
	sparsecol = (int*)malloc(2 * sizeof(*colno));
	for (j = 0; j < windowSize; j++) {
		for (i = 0; i < nodeSize; i++) {
			sparserow[0] = 1.0;
			sparsecol[0] = (1 + i) * windowSize + 1 + j;
			sparsecol[1] = windowSize * (nodeSize + 1) + 1 + j * nodeSize + i;
/*			if ((int)(job_array[j].min_cpus) < (int)(node_array[i].rem_cpus))
				sparserow[1] = -(int)job_array[j].min_cpus;
			else 
				sparserow[1] = -(int)node_array[i].rem_cpus;
*/
			sparserow[1] = -1.0*MIN((int)(job_array[j].min_cpus),(int)(node_array[i].rem_cpus));
			add_constraintex(lp, 2, sparserow, sparsecol, LE, 0);
			sparserow[0] = 1.0;
			sparsecol[0] = (1 + i) * windowSize + 1 + j;
			sparsecol[1] = windowSize * (nodeSize + 1) + 1 + j * nodeSize + i;
			sparserow[1] = -1;
			add_constraintex(lp, 2, sparserow, sparsecol, GE, 0);
		}
	}
	free(sparserow);
	free(sparsecol);

        /*
	cost is only selected nodes 
	c_j constraint; sum_i(t_ji) / (nodeSize + 1)
	*/
	sparserow = (REAL*)malloc((nodeSize + 1) * sizeof(*sparserow));
	sparsecol = (int*)malloc((nodeSize + 1) * sizeof(*colno));
	for (j = 0; j < windowSize; j++) {
		for (i = 0; i < nodeSize; i++) {
			sparserow[i] = 1.0;
			sparsecol[i] = windowSize * (nodeSize + 1) + 1 + j * nodeSize + i;
		}
		sparserow[nodeSize] = -(nodeSize + 1);
		sparsecol[nodeSize] = windowSize * (2 * nodeSize + 1) + j + 1;
		add_constraintex(lp, nodeSize + 1, sparserow, sparsecol, EQ, 0);
	}
	free(sparserow);
	free(sparsecol);

        /*
	min_nodes <= c_j * (nodeSize + 1) <= max_nodes
	*/
	sparserow = (REAL*)malloc((2) * sizeof(*sparserow));
	sparsecol = (int*)malloc((2) * sizeof(*colno));
	for (j = 0; j < windowSize; j++) {
		sparserow[0] = 1.0 + 1.0 * nodeSize;
		sparserow[1] = -(int)(job_array[j].min_nodes);
		sparsecol[0] = windowSize * (2 * nodeSize + 1) + 1 + j;
		sparsecol[1] = j + 1;
		add_constraintex(lp, 2, sparserow, sparsecol, GE, 0);
		sparserow[0] = 1.0 + 1.0 * nodeSize;
		sparserow[1] = -(int)(job_array[j].max_nodes);
		sparsecol[0] = windowSize * (2 * nodeSize + 1) + 1 + j;
		sparsecol[1] = j + 1;
		add_constraintex(lp, 2, sparserow, sparsecol, LE, 0);
	}
	free(sparserow);
	free(sparsecol);

	set_add_rowmode(lp, FALSE);
	/* set the objective function to p_j * (s_j - c_j) */
	for (j = 0; j < windowSize; j++) {
		colno[j] = j + 1;
		row[j] = (double)(job_array[j].priority);
		/* topo-aware 
		colno[j + windowSize] = j + 1 + windowSize;
		row[j + windowSize] = -job_array[j].priority;		
		*/
	}
	set_obj_fnex(lp, windowSize, row, colno);
	set_maxim(lp);
	/* 
	set the s_j to be binary
	x_ij to integer
	t_ij to binary
	*/	
	for (j = 1; j <= windowSize; j++) {
		set_int(lp, j, TRUE); /* s_j */
		set_bounds(lp, j, 0.0, 1.0);
	}
	for (j = windowSize + 1; j <= windowSize * (nodeSize + 1); j++) {
		set_int(lp, j, TRUE); /* x_ij */
		set_bounds(lp, j, 0.0, get_infinite(lp));
	}
	for (j = windowSize * (nodeSize + 1) + 1; j <= windowSize * (2 * nodeSize + 1); j++) {
		set_int(lp, j, TRUE); /* t_ij */
		set_bounds(lp, j, 0.0, 1.0);
	}
	set_verbose(lp,NORMAL);
	write_lp(lp, "model.lp");
	k = solve(lp);
	get_variables(lp, var);	
	delete_lp(lp);
	for (j = 0; j < windowSize; j++) {
		if (var[j] > 0) {
			solver_job_ptr = &job_array[j];
			solver_job_ptr->node_bitmap = (bitstr_t *) bit_alloc (node_record_count);
			solver_job_ptr->job_ptr->details->req_node_bitmap = (bitstr_t *) bit_alloc (node_record_count);
			solver_job_ptr->onnodes = (uint32_t *) xmalloc (sizeof(uint32_t)*node_record_count);/**/
			solver_job_ptr->job_ptr->details->req_node_layout = (uint16_t *)xmalloc(sizeof(uint16_t) * node_record_count);
			solver_job_ptr->job_ptr->details->req_node_bitmap = (bitstr_t *) bit_alloc (node_record_count);
			for (i = 0; i < nodeSize; i++) {
				k = (1 + i) * windowSize + j;
				if (var[k] > 0) {
					bit_set (solver_job_ptr->node_bitmap, (bitoff_t) (i));
					bit_set (solver_job_ptr->job_ptr->details->req_node_bitmap, (bitoff_t) (i));		
					node_array[i].rem_cpus -= var[k];
					node_array[i].rem_gpus -= solver_job_ptr->gpu;
					solver_job_ptr->onnodes[i] = var[k]; /**/
					solver_job_ptr->job_ptr->details->req_node_layout[i] = solver_job_ptr->onnodes[i]; 
					solver_job_ptr->alloc_total += var[k];
				}
			}
		} else
			job_array[j].alloc_total = 0;
	} 
	return k;
}

/* Try to start the job on any non-reserved nodes */
static int _start_job(struct job_record *job_ptr, bitstr_t *bitmap)
{
	int rc;
	bitstr_t *orig_exc_nodes = NULL;
	static uint32_t fail_jobid = 0;

	bit_not(bitmap);
	job_ptr->details->exc_node_bitmap = bit_copy(bitmap);
	rc = select_nodes(job_ptr, false, NULL);
	if (job_ptr->details) { /* select_nodes() might cancel the job! */
		FREE_NULL_BITMAP(job_ptr->details->exc_node_bitmap);
		job_ptr->details->exc_node_bitmap = orig_exc_nodes;
	} else
		FREE_NULL_BITMAP(orig_exc_nodes);
	if (rc == SLURM_SUCCESS) {
		/* job initiated */
		last_job_update = time(NULL);
		info("lpsched: Started JobId=%u on %s",
		     job_ptr->job_id, job_ptr->nodes);
		if (job_ptr->batch_flag == 0)
			srun_allocate(job_ptr->job_id);
		else if (job_ptr->details->prolog_running == 0)
			launch_job(job_ptr);
	} else if ((job_ptr->job_id != fail_jobid) &&
		   (rc != ESLURM_ACCOUNTING_POLICY)) {
		char *node_list;
		node_list = bitmap2node_name(bitmap);
		/* This happens when a job has sharing disabled and
		 * a selected node is still completing some job,
		 * which should be a temporary situation. */
		verbose("lpsched: Failed to start JobId=%u on %s: %s",
			job_ptr->job_id, node_list, slurm_strerror(rc));
		xfree(node_list);
		fail_jobid = job_ptr->job_id;
	} else {
		debug3("lpsched: Failed to start JobId=%u: %s",
		       job_ptr->job_id, slurm_strerror(rc));
	}

	return rc;
}


static int _run_solver_opt(void)
{
	bool filter_root = false;
	sched_nodeinfo_t* node_array = NULL;
	List job_queue;
	job_queue_rec_t *job_queue_rec;
	slurmdb_qos_rec_t *qos_ptr = NULL;
	char str[64];
	int i, j, solver_job_idx;
	struct job_record *job_ptr;
	solver_job_list_t *job_list, *solver_job_ptr;
	struct part_record *part_ptr = NULL;
	uint32_t end_time, time_limit, comp_time_limit, orig_time_limit;
	uint32_t min_nodes, max_nodes, req_nodes, minprio = (uint32_t)0;
	bitstr_t *avail_bitmap = NULL, *resv_bitmap = NULL;
	time_t now = time(NULL), sched_start, later_start, start_res;
	static int sched_timeout = 0;
	int this_sched_timeout = 0, rc = 0;
	sched_start = now;
	if (sched_timeout == 0) {
		sched_timeout = slurm_get_msg_timeout() / 2;
		sched_timeout = MAX(sched_timeout, 1);
		sched_timeout = MIN(sched_timeout, 10);
	}
	this_sched_timeout = sched_timeout;

	if (slurm_get_root_filter())
		filter_root = true;

	job_queue = build_job_queue(true);
	if (list_count(job_queue) < 1) {
		debug("sched: no jobs to run the solver");
		list_destroy(job_queue);
		return 0;
	}

	debug("generating job queue at %lld",(long long)sched_start);
	/* generate job queue */
	job_list = (solver_job_list_t*)malloc(max_job_count * sizeof(solver_job_list_t));
	/* will be converted to priority ordered list */
	solver_job_idx = 0;
	while ((job_queue_rec = (job_queue_rec_t *)
				list_pop_bottom(job_queue, sort_job_queue2))) {
		job_ptr  = job_queue_rec->job_ptr;
		part_ptr = job_queue_rec->part_ptr;
		xfree(job_queue_rec);
		if (!IS_JOB_PENDING(job_ptr))
			continue;	/* started in other partition */
		job_ptr->part_ptr = part_ptr;

		if ((job_ptr->state_reason == WAIT_ASSOC_JOB_LIMIT) ||
		    (job_ptr->state_reason == WAIT_ASSOC_RESOURCE_LIMIT) ||
		    (job_ptr->state_reason == WAIT_ASSOC_TIME_LIMIT) ||
		    (job_ptr->state_reason == WAIT_QOS_JOB_LIMIT) ||
		    (job_ptr->state_reason == WAIT_QOS_RESOURCE_LIMIT) ||
		    (job_ptr->state_reason == WAIT_QOS_TIME_LIMIT) ||
		    !acct_policy_job_runnable(job_ptr)) {
			debug2("lpsched: job %u is not allowed to run now. "
			       "Skipping it. State=%s. Reason=%s. Priority=%u",
			       job_ptr->job_id,
			       job_state_string(job_ptr->job_state),
			       job_reason_string(job_ptr->state_reason),
			       job_ptr->priority);
			continue;
		}

		if (((part_ptr->state_up & PARTITION_SCHED) == 0) ||
		    (part_ptr->node_bitmap == NULL))
		 	continue;
		if ((part_ptr->flags & PART_FLAG_ROOT_ONLY) && filter_root)
			continue;

		if ((!job_independent(job_ptr, 0)) ||
		    (license_job_test(job_ptr, time(NULL)) != SLURM_SUCCESS))
			continue;

		/* Determine minimum and maximum node counts */
		min_nodes = MAX(job_ptr->details->min_nodes,
				part_ptr->min_nodes);
		if (job_ptr->details->max_nodes == 0)
			max_nodes = part_ptr->max_nodes;
		else
			max_nodes = MIN(job_ptr->details->max_nodes,
					part_ptr->max_nodes);
		max_nodes = MIN(max_nodes, 500000);     /* prevent overflows */
		if (job_ptr->details->max_nodes)
			req_nodes = max_nodes;
		else
			req_nodes = min_nodes;
		if (min_nodes > max_nodes) {
			/* job's min_nodes exceeds partition's max_nodes */
			continue;
		}

		/* Determine job's expected completion time */
		if (job_ptr->time_limit == NO_VAL) {
			if (part_ptr->max_time == INFINITE)
				time_limit = 365 * 24 * 60; /* one year */
			else
				time_limit = part_ptr->max_time;
		} else {
			if (part_ptr->max_time == INFINITE)
				time_limit = job_ptr->time_limit;
			else
				time_limit = MIN(job_ptr->time_limit,
						 part_ptr->max_time);
		}
		comp_time_limit = time_limit;
		orig_time_limit = job_ptr->time_limit;
		if (qos_ptr && (qos_ptr->flags & QOS_FLAG_NO_RESERVE))
			time_limit = job_ptr->time_limit = 1;
		else if (job_ptr->time_min && (job_ptr->time_min < time_limit))
			time_limit = job_ptr->time_limit = job_ptr->time_min;

		/* Determine impact of any resource reservations */
		later_start = now;
 		FREE_NULL_BITMAP(avail_bitmap);
		start_res   = later_start;
		later_start = 0;
		j = job_test_resv(job_ptr, &start_res, true, &avail_bitmap);
		if (j != SLURM_SUCCESS) {
			job_ptr->time_limit = orig_time_limit;
			continue;
		}
		if (start_res > now)
			end_time = (time_limit * 60) + start_res;
		else
			end_time = (time_limit * 60) + now;

		solver_job_ptr = &job_list[solver_job_idx++];
		solver_job_ptr->job_ptr = job_ptr;
		solver_job_ptr->job_id = job_ptr->job_id;
		solver_job_ptr->min_nodes = min_nodes;
		solver_job_ptr->max_nodes = min_nodes;
		solver_job_ptr->gpu = gres_job_gpu_count(solver_job_ptr->job_ptr->gres_list);
		solver_job_ptr->min_cpus = job_ptr->details->min_cpus;
		solver_job_ptr->max_cpus = job_ptr->details->max_cpus;
		solver_job_ptr->priority = job_ptr->priority;
		if ((double)solver_job_ptr->priority < (double)minprio)
			minprio = (double)solver_job_ptr->priority;
		debug("minprio: %d %f",minprio,(double)minprio);
		if ((int)solver_job_ptr->max_cpus < (int)solver_job_ptr->min_cpus)
			solver_job_ptr->max_cpus = solver_job_ptr->min_cpus;
	}
	for (i = 0; i < solver_job_idx; i++) {
		job_list[i].priority -= (minprio - 1);
		debug("prio before solve %d: %f - %u",i,(double)(job_list[i].priority),(job_list[i].priority));
	}
	node_array = _print_nodes_inlpsched();
	solve_allocation(node_record_count, solver_job_idx, 3, node_array, job_list);
	debug("xafter allocation, heres the result:");
	for (i=0;i<node_record_count;i++)
		debug("node %d remgpu %u remcpu %u",i,node_array[i].rem_gpus,node_array[i].rem_cpus);

	for (i=0;i<solver_job_idx;i++) {
		solver_job_ptr = &job_list[i];
		job_ptr = solver_job_ptr->job_ptr;
		debug("job %d requirements, minnodes: %d, maxnodes: %d, mincpus: %d, maxcpus: %d, allocated: %d",
			solver_job_ptr->job_id, solver_job_ptr->min_nodes, solver_job_ptr->max_nodes, 
			solver_job_ptr->min_cpus, solver_job_ptr->max_cpus, solver_job_ptr->alloc_total);
		if (!solver_job_ptr->alloc_total) { 
			continue;
		} /* job is allocated resources, create req_node_layout matrix */
		bit_and(avail_bitmap, solver_job_ptr->node_bitmap); 		
		/* Identify usable nodes for this job */
		bit_and(avail_bitmap, part_ptr->node_bitmap);
		bit_and(avail_bitmap, up_node_bitmap);

		if (job_ptr->details->exc_node_bitmap) {
			bit_not(job_ptr->details->exc_node_bitmap);
			bit_and(avail_bitmap,
				job_ptr->details->exc_node_bitmap);
			bit_not(job_ptr->details->exc_node_bitmap);
		}

		/* Identify nodes which are definitely off limits */
		FREE_NULL_BITMAP(resv_bitmap);
		resv_bitmap = bit_copy(avail_bitmap);
		bit_not(resv_bitmap);

		if ((time(NULL) - sched_start) >= this_sched_timeout) {
			debug("lpsched: loop taking too long, yielding locks");
			if (_yield_locks()) {
				debug("lpsched: system state changed, "
				      "breaking out");
				rc = 1;
				break;
			} else {
				this_sched_timeout += sched_timeout;
			}
		}
		_start_job(job_ptr, solver_job_ptr->node_bitmap);
	}

	FREE_NULL_BITMAP(avail_bitmap);
	FREE_NULL_BITMAP(resv_bitmap);

	list_destroy(job_queue);
	return rc;
}

/*
serenkod
*/
struct sched_nodeinfo* _print_nodes_inlpsched()
{
	time_t d = time(NULL);
	int i;
	struct select_nodeinfo *nodeinfo = NULL;
	struct sched_nodeinfo* node_array = xmalloc(sizeof(struct sched_nodeinfo* ) * node_record_count);
	info("currently in _print_nodes_lpsched();");
	select_g_select_nodeinfo_set_all(d);
	for (i = 0; i < node_record_count; i++) {
		select_g_select_nodeinfo_get(node_record_table_ptr[i].select_nodeinfo,
					     SELECT_NODEDATA_PTR, 0,
					     (void *)&nodeinfo);
		if(!nodeinfo) {
			error("no nodeinfo returned from structure");
			continue;
		}
		node_array[i].rem_gpus = nodeinfo->rem_gpus;
		node_array[i].rem_cpus = nodeinfo->rem_cpus;
		/* 
		test commands
		info("node %d cpu %u",i,node_record_table_ptr[i].cpus);
		info("lpschedde alloc cpu in node %d is %u, rem_gpu %u",i,nodeinfo->alloc_cpus,nodeinfo->rem_gpus); 
		*/
	}
	return node_array;
}


