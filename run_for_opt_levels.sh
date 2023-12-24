n=$1
k=$2

for i in `seq 0 9`; do
	MULT=1 OPT=$i LOG=0 make -B
	./main $n $k 3 23850
done
