class FibTask: public task {
public:
	const long n;
	long* const sum;
	FibTask( long n_, long* sum_ ) :
		n(n_), sum(sum_)
	{}
	task* execute() {	//Overrieds virtual function task:: execute
		if( n<CutOff) {
			*sum = SerialFib(n);
		} else {
			long x, y;
			FibTask& a = *new( allocate_child() ) FibTask(n-1, &x);
			FibTask& b = *new( allocate_child() ) FibTask(n-2, &y);
			//Set ref_count to "two children plus one for the wait".
			set_ref_count(3);
			//start b
			spawn(b);
			//Start a and wait for children
			spawn_and_wait_for_all(a);
			//Do the sum
			*sum = x+y;
		}
		return NULL;
	}
};