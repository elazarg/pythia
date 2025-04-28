

.PHONY: all

all: diff_coredump diff_vm_snapshot

diff_coredump:
	gcc -O2 -std=c17 diff_coredump.c -o diff_coredump
	

diff_vm_snapshot:
	g++ -O2 -std=c++20 diff_vm_snapshot.cpp -o diff_vm_snapshot

clean:
	rm diff_coredump diff_vm_snapshot

