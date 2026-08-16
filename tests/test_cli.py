import subprocess,sys
def test_help():
    result=subprocess.run([sys.executable,'-m','qkernel','--help'],text=True,capture_output=True)
    assert result.returncode==0 and 'usage:' in result.stdout
def test_cli_rejects_unknown_argument():
    result=subprocess.run([sys.executable,'-m','qkernel','--unknown-option'],text=True,capture_output=True)
    assert result.returncode!=0

def test_documented_workflow():
    import json
    result=subprocess.run([sys.executable,'-m','qkernel',*['examples/clusters.json']],text=True,capture_output=True)
    assert result.returncode==0, result.stderr
    data=json.loads(result.stdout)
    assert data['test_accuracy']==1 and data['baseline_accuracy']==1
