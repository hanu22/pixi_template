import spacy


@spacy.registry.callbacks("custom_callback")
def create_custom_callback():
    def custom_callback():
        return 256

    return custom_callback
