# Makefile for backfill scheduler plugin

AUTOMAKE_OPTIONS = foreign

CXXFLAGS = -fexceptions

PLUGIN_FLAGS = -module -avoid-version --export-dynamic -lm -ldl

INCLUDES = -I$(top_srcdir) -I$(top_srcdir)/src/common

pkglib_LTLIBRARIES = sched_lp.la

sched_lp_la_SOURCES = lpsched_wrapper.c	\
			lpsched.c	\
			liblpsolve55.so
			lpsched.h
sched_lp_la_LDFLAGS = $(SO_LDFLAGS) $(PLUGIN_FLAGS)
