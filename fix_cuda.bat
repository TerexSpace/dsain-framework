@echo off
REM Fix CUDA/PyTorch Installation
REM This will uninstall CPU-only PyTorch and install CUDA version

echo ========================================
echo Fixing PyTorch CUDA Installation
echo ========================================
echo.
echo Current status:
echo   - GPU: RTX 4060 (Detected, Working)
echo   - CUDA: 13.0 (Installed)
echo   - PyTorch: CPU-only (WRONG!)
echo.
echo This will:
echo   1. Uninstall CPU-only PyTorch
echo   2. Install PyTorch with CUDA support
echo.
echo ========================================

pause

echo.
echo Step 1: Uninstalling CPU-only PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Step 2: Installing PyTorch with CUDA 12.4 support...
echo (Compatible with your CUDA 13.0)
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo ========================================
echo Verifying installation...
echo ========================================
python -c "import torch; print(f'\nPyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not detected\"}'); print('\n✅ SUCCESS! CUDA is now available!' if torch.cuda.is_available() else '❌ FAILED - CUDA still not available')"

echo.
echo ========================================
pause
