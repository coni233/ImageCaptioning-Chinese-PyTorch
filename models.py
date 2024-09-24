import torch
import torchvision
from torch import nn

from torchviz import make_dot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        #resnet = torchvision.models.resnet101(pretrained=True)  # 表示加载一个在 ImageNet 数据集上预训练的模型。
        
        # resnet = torchvision.models.resnet101(pretrained=False) # 这里只是我用本地的路径来加载模型
        # model_weights = torch.load('resnet101-cd907fc2.pth')
        # resnet.load_state_dict(model_weights)

        # 使用新的 weights 参数
        '''获取的预训练权重通常是在 ImageNet-1K 数据集上训练得到的。
        这个数据集是 ImageNet 数据库中图像分类任务的一个子集，
        包含了大约 1000 个类别，每个类别大约有 1000 张标注图像，
        总共有超过一百万张图片。'''
        weights = torchvision.models.ResNet101_Weights.DEFAULT
        resnet = torchvision.models.resnet101(weights=weights)


        # 移除线性层和池化层（因为我们不做分类）
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # 调整图像大小以允许输入不同大小的图像
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        前向传播
        :param images: 图像张量，维度为 (batch_size, 3, image_size, image_size)
        :return: 编码后的图像
        """
        images = images.to(device)  # 确保输入张量在同一个设备上
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        允许或禁止对编码器的卷积块 2 到 4 进行梯度计算。
        :param fine_tune: 允许？
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: 编码图像的特征大小
        :param decoder_dim: 解码器 RNN 的大小
        :param attention_dim: 注意力网络的大小
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # 线性层，用于转换编码图像
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # 线性层，用于转换解码器的输出
        self.full_att = nn.Linear(attention_dim, 1)  # 线性层，用于计算 softmax 的值
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax 层，用于计算权重

    def forward(self, encoder_out, decoder_hidden):
        """
        前向传播
        :param encoder_out: 编码图像，维度为 (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: 上一个时间步解码器的输出，维度为 (batch_size, decoder_dim)
        :return: 注意力加权编码，权重
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: 注意力网络的大小
        :param embed_dim: 嵌入层大小
        :param decoder_dim: 解码器 RNN 的大小
        :param vocab_size: 词汇表大小
        :param encoder_dim: 编码图像的特征大小2048对应resnet101的输出
        :param dropout: dropout 概率
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # 解码 LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # 线性层，用于找到 LSTMCell 的初始隐藏状态
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # 线性层，用于找到 LSTMCell 的初始细胞状态
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # 线性层，用于创建一个 sigmoid 激活的门
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # 线性层，用于找到词汇表上的分数
        self.init_weights()  # 用均匀分布初始化一些层

    def init_weights(self):
        """
        布初始化一些参数，以便更容易收敛。
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        加载预训练的嵌入层。
        :param embeddings: 预训练的嵌入层
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        允许微调嵌入层？（如果使用预训练的嵌入，这才有意义）。
        :param fine_tune: 允许？
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        基于编码图像创建解码器 LSTM 的初始隐藏和细胞状态。
        :param encoder_out: 编码图像，维度为 (batch_size, num_pixels, encoder_dim)
        :return: 隐藏状态，细胞状态
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        前向传播
        :param encoder_out: 编码图像，维度为 (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: 编码字幕，维度为 (batch_size, max_caption_length)
        :param caption_lengths: 字幕长度，维度为 (batch_size, 1)
        :return: 词汇表的分数，排序后的编码字幕，解码长度，权重，排序索引
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # 展平图像
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # 按长度递减排序输入数据；为什么？下面会解释
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # 初始化 LSTM 状态
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # 我们不会在 <end> 位置解码，因为一旦生成 <end> 就完成了生成
        # 所以，解码长度是实际长度 - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # 创建张量以保存词预测分数和 alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # 在每个时间步，通过
        # 基于解码器的上一个隐藏状态输出对编码器的输出进行注意力加权
        # 然后用上一个词和注意力加权编码在解码器中生成一个新词
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # 门控标量, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


if __name__ == '__main__':
    encoder = Encoder()
    decoder = DecoderWithAttention(attention_dim=512,
                                   embed_dim=512,
                                   decoder_dim=512,
                                   vocab_size=10720)
    encoder.to(device)
    decoder.to(device)
    print(encoder)
    print(decoder)

    # 在yolov3-ic环境中运行这个，还得改
    # 生成encoder
    # 创建模型实例并移动到设备
    model_encoder = encoder.to(device)   
    example_input_encoder = torch.randn(1, 3, 224, 224).to(device)
    output_encoder = model_encoder(example_input_encoder)
    # 生成并保存图形
    dot_encoder = make_dot(output_encoder, params=dict(model_encoder.named_parameters())) #这种方式不包括输入数据的显示，仅仅显示模型的参数
    #dot = make_dot(output, params=dict(list(encoder.named_parameters()) + [('input', example_input)])) #通过将 example_input 明确地作为参数传入，它能让你在计算图中看到输入数据的节点
    dot_encoder.render('encoder_model_visualization', format='svg')  # 保存为 svg 文件
    
    # 生成decoder,还有问题,这个还得改
    # model_decoder = decoder.to(device)
    # encoder_out = torch.randn(1, 196, 2048).to(device) # 模拟编码后的图像输出
    # encoded_captions = torch.randint(0, 1000, (2, 10))  # 模拟的字幕
    # caption_lengths = torch.LongTensor([10] * 2)  # 模拟的字幕长度
    # caption_lengths = caption_lengths.unsqueeze(1)
    # print(caption_lengths.shape)
    # caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
    # encoder_out = encoder_out.to(device)
    # encoded_captions = encoded_captions.to(device)
    # caption_lengths = caption_lengths.to(device)
    # outputs_decoder = model_decoder(encoder_out, encoded_captions, caption_lengths)
    # preds = outputs_decoder[0]
    # dot_decoder = make_dot(preds, params=dict(model_decoder.named_parameters())) #这种方式不包括输入数据的显示，仅仅显示模型的参数
    # dot_decoder.render('decoder_model_visualization', format='svg')  # 保存为 svg 文件

