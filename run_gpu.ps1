$ErrorActionPreference = 'Stop'
$windowsProject = $PSScriptRoot.Replace('\', '/')
$linuxProject = & wsl -d Ubuntu-24.04 -u root --exec wslpath -a $windowsProject
if ($LASTEXITCODE -ne 0) { throw 'Não foi possível acessar o projeto pelo WSL2.' }
$linuxProject = $linuxProject.Trim()
& wsl -d Ubuntu-24.04 -u root --cd $linuxProject -- env TRAINING_BACKEND=gpu /opt/final-project-gpu/bin/python -u main.py @args
exit $LASTEXITCODE
