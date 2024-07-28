#!/usr/bin/env bash
# install qemu-9.0.0 with:
# ./configure --target-list=x86_64-softmmu --enable-virtfs --disable-glusterfs --disable-seccomp --disable-{bzip2,snappy,lzo} --disable-usb-redir --disable-libusb --disable-libnfs  --disable-libiscsi --disable-rbd --disable-spice --disable-cap-ng --disable-linux-aio --disable-brlapi --disable-vnc-{jpeg,sasl} --disable-rdma --disable-curl --disable-curses --disable-sdl --disable-gtk  --disable-tpm --disable-vte --disable-vnc  --disable-xen --disable-opengl
# make -j$(nproc)

# experiment is the first argument
EXPERIMENT=$1
if [ -z "$EXPERIMENT" ]; then
  echo "Usage: $0 <experiment>"
  exit 1
fi
PATH_TO_SHARE="./experiment"
if [ ! -d "$PATH_TO_SHARE" ]; then
  echo "Directory $PATH_TO_SHARE does not exist."
  exit 1
fi
shift 1

# This is already in qcow2 format.
# https://cloud-images.ubuntu.com/minimal/releases/noble/release/ubuntu-24.04-minimal-cloudimg-amd64.img
img=ubuntu-24.04-minimal-cloudimg-amd64.img
if [ ! -f "./$img" ]; then
  wget "https://cloud-images.ubuntu.com/minimal/releases/noble/release/${img}"
  ${QEMU_DIR}qemu-img resize ${img} +2G
fi

INSTANCE_DIR="./pool/${EXPERIMENT}"
rm -rf ${INSTANCE_DIR}
mkdir -p ${INSTANCE_DIR}

instance="${INSTANCE_DIR}/vm.img"
cp ./${img} ${instance}

MOUNT_TAG="experiment"
TARGET_DIR="/home/ubuntu/experiment"

yaml_file="${INSTANCE_DIR}/user-data.yaml"
cat > ${yaml_file} <<EOF
#cloud-config

# For the password.
# user: "ubuntu"
password: asdfqwer
chpasswd: { expire: False }
ssh_pwauth: True

packages:
  - python3-pip

package_update: true
package_upgrade: true
package_reboot_if_required: true
allow_public_ssh_keys: true
mounts:
 - [${MOUNT_TAG}, ${TARGET_DIR}, 9p]

write_files:
  - path: /home/ubuntu/.bashrc
    content: |+
      export PYTHONPATH=/home/ubuntu/
    append: true

runcmd:
  - [su, ubuntu, -c, "pip3 install -r ${TARGET_DIR}/requirements.txt"]
  - [su, ubuntu, -c, "pip3 install --break-system-packages -r ${TARGET_DIR}/${EXPERIMENT}/requirements.txt"]
EOF

user_data="${INSTANCE_DIR}/user-data.qcow2"
cloud-localds ${user_data} ${yaml_file} --disk-format=qcow2

args=(
  -cpu host
  -smp 1
  -drive "file=${instance},format=qcow2"
  -drive "file=${user_data},format=qcow2"
  -enable-kvm
  -m 2G
#  -serial mon:stdio  # use console for monitor
  -qmp tcp:localhost:4444,server=on,wait=off
#  -nic user
  -device virtio-net-pci,netdev=n1
  -netdev user,id=n1,hostfwd=tcp::10022-:22
  -virtfs local,path=${PATH_TO_SHARE},mount_tag=${MOUNT_TAG},security_model=none
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
