from transformers import AutoModel, AutoTokenizer

class PretrainedModelLoader:
    def __init__(self, config):
        self.config = config

    def load_model(self, model_name):
        return AutoModel.from_pretrained(model_name)

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def get_model_and_tokenizer(self, model_name):
        model = self.load_model(model_name)
        tokenizer = self.load_tokenizer(model_name)
        return model, tokenizer

class PretrainedModelManager:
    def __init__(self, config):
        self.config = config
        self.loader = PretrainedModelLoader(config)
        self.loaded_models = {}

    def get_model(self, model_name):
        if model_name not in self.loaded_models:
            model, tokenizer = self.loader.get_model_and_tokenizer(model_name)
            self.loaded_models[model_name] = (model, tokenizer)
        return self.loaded_models[model_name]

    def unload_model(self, model_name):
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]

    def get_available_models(self):
        # This could be expanded to check for locally available models or query an API
        return list(self.config['pretrained_models'].keys())