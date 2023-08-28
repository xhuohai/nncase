import nncase

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def compile_kmodel(model_path):
    import_options = nncase.ImportOptions()
    compile_options = nncase.CompileOptions()
    compile_options.target = "cpu"
    compiler = nncase.Compiler(compile_options)
    model_content = read_model_file(model_path)
    if model_path.split(".")[-1] == "onnx":
        compiler.import_onnx(model_content, import_options)
    elif model_path.split(".")[-1] == "tflite":
        compiler.import_tflite(model_content, import_options)
    compiler.compile()
    compiler.gencode_tobytes()

# test failed when the conf in toml but do not set the default value of compile option
def test_compile_option_default_value():
    model_path = "../../examples/user_guide/test.onnx"
    kmodel_path = compile_kmodel(model_path)

if __name__ == "__main__":
    pytest.main(['-vv', 'test_compile_option_default_value'])
