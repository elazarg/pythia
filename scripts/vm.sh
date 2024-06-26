#!/usr/bin/env bash
# install qemu-9.0.0 with:
# ./configure --target-list=x86_64-softmmu --disable-glusterfs --disable-seccomp --disable-{bzip2,snappy,lzo} --disable-usb-redir --disable-libusb --disable-libnfs  --disable-libiscsi --disable-rbd --disable-spice --disable-attr --disable-cap-ng --disable-linux-aio --disable-brlapi --disable-vnc-{jpeg,sasl} --disable-rdma --disable-curl --disable-curses --disable-sdl --disable-gtk  --disable-tpm --disable-vte --disable-vnc  --disable-xen --disable-opengl
# make -j$(nproc)

POOL=pool
# This is already in qcow2 format.
# https://cloud-images.ubuntu.com/minimal/releases/noble/release/ubuntu-24.04-minimal-cloudimg-amd64.img
img=ubuntu-24.04-minimal-cloudimg-amd64.img
if [ ! -f "$POOL/$img" ]; then
  cd $POOL
  wget "https://cloud-images.ubuntu.com/minimal/releases/noble/release/${img}"
  ${QEMU_DIR}qemu-img resize ${img} +2G
  cd ..
fi

user_data=$POOL/user-data.qcow2
if [ ! -f "$user_data" ]; then
  cloud-localds ${user_data} $POOL/user-data.yaml --disk-format=qcow2
fi

# run:
# sudo apt update
# sudo apt -y upgrade
# sudo apt -y install python3-pip python3.12-venv
# git clone https://github.com/elazarg/pythia
# cd pythia
# python3.12 -m venv venv
# source venv/bin/activate
# pip install -r experiment/requirements.txt -r experiment/{experiment}/requirements.txt

args=(
  -cpu host
  -smp 1
  -drive "file=${POOL}/${img},format=qcow2"
  -drive "file=${user_data},format=qcow2"
  -enable-kvm
  -m 2G
#  -serial mon:stdio  # use console for monitor
  -qmp tcp:localhost:4444,server=on,wait=off
#  -nic user
  -device virtio-net-pci,netdev=n1
  -netdev user,id=n1,hostfwd=tcp::10022-:22
#  -nographic
#  -display none
#  -daemonize
)
${QEMU_DIR}qemu-system-x86_64 "${args[@]}" "$@"
# args=(
#   --name nvram-vm
#   --cpu host
#   --vcpus 1
#   --ram 2048
#   --disk path=${img},format=qcow2
#   --disk path=${user_data},device=cdrom
#   --network network=default,model=rtl8139
#   --graphics none
#   --serial tcp,host=4444,protocol.type=telnet
#   --noautoconsole
#   --import
#   --os-variant ubuntu-stable-latest
# )
# virt-install "${args[@]}"
