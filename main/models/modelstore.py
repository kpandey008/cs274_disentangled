from models.vae import CIFAR10Encoder, CIFAR10Decoder, MNISTDecoder, MNISTEncoder, CelebaDecoder, CelebaEncoder


def get_model(name, config):
    code_size = config.model.code_size
    if name == 'cifar10':
        enc = CIFAR10Encoder(code_size, config.data.in_channels, config.model.encoder.base_ch, config.model.encoder.channel_mults)
        dec = CIFAR10Decoder(code_size)
    elif name == 'mnist':
        enc = MNISTEncoder(code_size, config.data.in_channels, config.model.encoder.base_ch, config.model.encoder.channel_mults)
        dec = MNISTDecoder(code_size)
    elif name == 'celeba':
        enc = CelebaEncoder(code_size, config.data.in_channels, config.model.encoder.base_ch, config.model.encoder.channel_mults)
        dec = CelebaDecoder(code_size)
    else:
        raise NotImplementedError()

    return enc, dec
