class ChannelGradCAM(object):
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        
        