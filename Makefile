
main: main.cu num primes.list
	@# suppress error 2464 (string->*char deprecated error)
	nvcc -o main main.cu -Igmp-6.3.0/build/include/ -Lgmp-6.3.0/build/lib -lgmp -lcuda -Xcudafe --display_error_number -Xcudafe --diag_suppress=2464 -O3 -D OPT=${OPT} -D BLOCK_MULT=${MULT}
