param(
    [int]$Port = 8000,
    [string]$Task = "real_sr",
    [int]$Scale = 4,
    [string]$Model = "model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir

try {
    $arguments = @(
        "run",
        "--no-capture-output",
        "-n",
        "torch",
        "python",
        "web_server.py",
        "--task",
        $Task,
        "--scale",
        $Scale,
        "--model_path",
        $Model,
        "--port",
        $Port,
        "--host",
        "0.0.0.0"
    )

    & conda @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
