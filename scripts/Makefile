

.PHONY: all

all: diff_coredump diff_vm_snapshot

diff_coredump:
	gcc -O2 -std=c17 diff_coredump.c -o diff_coredump
	

diff_vm_snapshot:
	g++ -O2 -std=c++20 diff_vm_snapshot.cpp -o diff_vm_snapshot

snapshot.so: snapshot_homegrown.c
	# Create shared library for Python
	gcc -shared -fPIC -O2 -fno-stack-protector -fno-ident -fmerge-all-constants -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-exceptions -o snapshot.so snapshot_homegrown.c

homegrown_debug: snapshot_homegrown.c
	# Debug version with extra checks
	gcc -DTEST_MAIN -g -DDEBUG -o debug_snapshot snapshot_homegrown.c

test_snapshot: snapshot_homegrown.c
	gcc -DTEST_MAIN -O2 -fno-stack-protector -fno-ident -fmerge-all-constants -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-exceptions -o test_snapshot snapshot_homegrown.c

clean:
	rm diff_coredump diff_vm_snapshot
