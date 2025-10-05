# Configure the git credentials
git config --global user.name "William Pan"
git config --global user.email "williampan4032@gmail.com"

# Configure default editor to vim
echo 'export EDITOR=vim' >> ~/.bashrc
source ~/.bashrc

# Install clangd
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 19
sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100
sudo apt install -y bear