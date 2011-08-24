//Serial
long SerialFib( long n ) {
	if( n<2 )
		return n;
	else
		return SerialFib(n-1)+SerialFib(n-2);
}

//Parallel
long ParallelFib( long n ) {
	long sum;
	FibTask& a = *new(task::allocate_root()) FibTask(n,&sum);
	task::spawn_root_and_wait(a);
	return sum;
}