

.PHONY: all

all: diff_coredump diff_vm_snapshot

diff_coredump:
	gcc -O2 -std=c17 diff_coredump.c -o diff_coredump
	

diff_vm_snapshot:
	g++ -O2 -std=c++20 diff_vm_snapshot.cpp -o diff_vm_snapshot

snapshot.so: homegrown_snapshot.c
	# Create shared library for Python
	gcc -shared -fPIC -O2 -fno-stack-protector -fno-ident -fmerge-all-constants -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-exceptions -o snapshot.so homegrown_snapshot.c

homegrown_debug: homegrown_snapshot.c
	# Debug version with extra checks
	gcc -DTEST_SNAPSHOT -g -DDEBUG -o debug_snapshot homegrown_snapshot.c

test_snapshot: homegrown_snapshot.c
	gcc -DTEST_SNAPSHOT -O2 -fno-stack-protector -fno-ident -fmerge-all-constants -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-exceptions -o test_snapshot homegrown_snapshot.c

clean:
	rm -f snapshot.so debug_snapshot test_snapshot
	rm -f diff_coredump diff_vm_snapshot
