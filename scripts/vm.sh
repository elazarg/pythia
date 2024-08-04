#!/usr/bin/env bash
# install qemu-9.0.0 with:
# ./configure --target-list=x86_64-softmmu --enable-virtfs --disable-glusterfs --disable-seccomp --disable-{bzip2,snappy,lzo} --disable-usb-redir --disable-libusb --disable-libnfs  --disable-libiscsi --disable-rbd --disable-spice --disable-cap-ng --disable-linux-aio --disable-brlapi --disable-vnc-{jpeg,sasl} --disable-rdma --disable-curl --disable-curses --disable-sdl --disable-gtk  --disable-tpm --disable-vte --disable-vnc  --disable-xen --disable-opengl
# make -j$(nproc)

# experiment is the first argument
EXPERIMENT=$1
if [ -z "$EXPERIMENT" ]; then
  echo "Usage: $0 <experiment> [QMP_PORT] [TCP_PORT] [STEP]"
  exit 1
fi

mkdir -f results
mkdir -f results/${EXPERIMENT}
chmod -R a+w results

STEP=${2:-1}
QMP_PORT=${3:-4444}
TCP_PORT=${4:-1234}
shift 4

EXPERIMENT_PATH="experiment/${EXPERIMENT}"
if [ ! -d "$EXPERIMENT_PATH" ]; then
  echo "Directory $EXPERIMENT_PATH does not exist."
  exit 1
fi

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

PROJECT_DIR="/mnt/pythia"
GUEST_HOME="/home/ubuntu"
VENV_BIN="${GUEST_HOME}/.venv/bin"

yaml_file="${INSTANCE_DIR}/user-data.yaml"
cat > ${yaml_file} <<EOF
#cloud-config

# For the password.
# user: "ubuntu"
password: asdfqwer
chpasswd: { expire: False }
ssh_pwauth: True
allow_public_ssh_keys: true

packages:
  - python3-pip
  - python3-venv

package_update: false
package_upgrade: false

mounts:
 - [pythia, ${PROJECT_DIR}, 9p]

write_files:
  - path: ${GUEST_HOME}/.bashrc
    permissions: '0640'
    owner: ubuntu:ubuntu
    defer: true
    append: true
    content: |+
      export EXPERIMENT=${EXPERIMENT}
      export STEP=${STEP}
      source ${VENV_BIN}/activate
      export PYTHONPYCACHEPREFIX=/tmp/pythia
      export PYTHONPATH=${PROJECT_DIR}
      cd ${PROJECT_DIR}

  - path: ${GUEST_HOME}/.bash_history
    owner: ubuntu:ubuntu
    defer: true
    content: |+
      cat ${EXPERIMENT_PATH}/args.txt | xargs python ${EXPERIMENT_PATH}/naive.py
      cat ${EXPERIMENT_PATH}/args.txt | xargs python ${EXPERIMENT_PATH}/instrumented.py
      cat ${EXPERIMENT_PATH}/args.txt | xargs python ${EXPERIMENT_PATH}/vm.py

runcmd:
  - sudo chown -R ubuntu:ubuntu ${GUEST_HOME}
  - [su, ubuntu, -c, "python3 -m venv ${GUEST_HOME}/.venv"]
  - [su, ubuntu, -c, "${VENV_BIN}/pip install -r ${PROJECT_DIR}/checkpoint/requirements.txt"]
  - [su, ubuntu, -c, "${VENV_BIN}/pip install -r ${PROJECT_DIR}/${EXPERIMENT_PATH}/requirements.txt"]
EOF

user_data="${INSTANCE_DIR}/user-data.qcow2"
cloud-localds ${user_data} ${yaml_file} --disk-format=qcow2

args=(
  -cpu host
  -smp 1
  -drive "file=${instance},format=qcow2"
  -drive "file=${user_data},format=qcow2"
  -virtfs local,path=".",mount_tag=pythia,security_model=mapped
  -enable-kvm
  -m 2G
#  -serial mon:stdio  # use console for monitor
  -qmp tcp:localhost:${QMP_PORT},server=on,wait=off
#  -nic user
  -device virtio-net-pci,netdev=n1
  -netdev user,id=n1,hostfwd=tcp::10022-:22
  -nographic
#  -display none
#  -daemonize
)

${QEMU_DIR}qemu-system-x86_64 "${args[@]}"
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
