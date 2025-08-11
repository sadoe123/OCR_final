@echo off
echo ================================
echo === Compilation en cours... ===
echo ================================

REM Nettoyage des anciens fichiers
rmdir /s /q dist
rmdir /s /q build
del /q *.spec

REM Création du nouvel .exe
pyinstaller final.spec  REM Utilisation du fichier .spec




echo ================================
echo === Compilation terminée ! ===
echo === Fichier disponible dans dist\ ===
echo ================================
pause
