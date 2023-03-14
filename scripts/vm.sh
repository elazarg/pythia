#!/usr/bin/env bash

# This is already in qcow2 format.
img=ubuntu-22.10-server-cloudimg-amd64.img
if [ ! -f "$img" ]; then
  wget "https://cloud-images.ubuntu.com/releases/22.10/release/${img}"

  # sparse resize: does not use any extra space, just allows the resize to happen later on.
  # https://superuser.com/questions/1022019/how-to-increase-size-of-an-ubuntu-cloud-image
  qemu-img resize "$img" + 128G
fi

user_data=user-data.qcow2
if [ ! -f "$user_data" ]; then
  yaml_file=user-data.yaml
  # For the password.
  # user: "ubuntu"
  # https://stackoverflow.com/questions/29137679/login-credentials-of-ubuntu-cloud-server-image/53373376#53373376
  # https://serverfault.com/questions/920117/how-do-i-set-a-password-on-an-ubuntu-cloud-image/940686#940686
  # https://askubuntu.com/questions/507345/how-to-set-a-password-for-ubuntu-cloud-images-ie-not-use-ssh/1094189#1094189
  cat >${yaml_file} <<EOF
#cloud-config
password: asdfqwer
chpasswd: { expire: False }
ssh_pwauth: True
EOF
  cloud-localds ${user_data} ${yaml_file} --disk-format=qcow2
fi

args=(
  -cpu host
  -smp 1
  -drive "file=${img},format=qcow2"
  -drive "file=${user_data},format=qcow2"
#  -device rtl8139,netdev=net0
  -enable-kvm
  -m 2G
#  -netdev user,id=net0
   -serial mon:stdio  # use console for monitor
  # -chardev socket,id=monitor,host=127.0.0.1,port=4444,server=on,wait=off,telnet=on
#   -mon chardev=monitor,mode=readline
  -qmp tcp:localhost:4444,server=on,wait=off
   -nographic
#  -daemonize
#  -net user,hostfwd=tcp::10022-:22
  -net nic
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
