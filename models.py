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


        # 移除线性层和池化层（因为我们只需要特征图，不需要分类的结果）
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # 建了一个自适应平均池化层，自适应平均池化层的作用是将输入特征图调整为指定的大小，无论输入特征图的原始大小如何
        # 就是在这里把图像分成了14*14个块，在下面attendion中，每个块都会被注意力机制关注
        # 这里合起来就是完整的图像，分开来就是每个块，真是巧妙啊！！！
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
        #重新排列张量的维度
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    #微调编码器的卷积块 2 到 4
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
        self.full_att = nn.Linear(attention_dim, 1)  # 线性层，用于计算 softmax 的值，计算注意力权重
        self.relu = nn.ReLU() # ReLU 激活函数
        self.softmax = nn.Softmax(dim=1)  # softmax 层，用于计算权重

    #计算编码图像和解码器输出的注意力权重，并生成注意力加权编码
    def forward(self, encoder_out, decoder_hidden):
        """
        前向传播
        :param encoder_out: 编码图像，维度为 (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: 解码器的隐藏状态，上一个时间步解码器的输出，维度为 (batch_size, decoder_dim)
        :return: 注意力加权编码，权重
        """
        #在前向传播过程中，首先将编码图像和解码器隐藏状态经过线性层变换得到注意力权重的两个部分
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim) 将编码图像的特征转换为注意力维度 （batch_size,196,2048） -> (batch_size,196,512)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim) 将解码器的输出转换为注意力维度 （batch_size,128） -> (batch_size,512,1)
        #接下来，通过多层感知机的方式计算注意力权重
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # 计算注意力权重，将两个部分相加，并经过 ReLU 激活函数和线性层变换，得到一个维度为 (batch_size, num_pixels) 的向量
        alpha = self.softmax(att)  # 使用 softmax 函数将该向量进行归一化，得到注意力权重，维度为 (batch_size, num_pixels)
        #最后，根据注意力权重对编码图像进行加权求和，得到注意力加权的编码向量
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim) 计算注意力加权编码

        #最终的输出包括注意力加权的编码向量和注意力权重。
        return attention_weighted_encoding, alpha


#通过注意力机制，解码器能够在生成每个单词时动态地关注输入图像的不同区域，从而生成更准确和连贯的描述
class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: 注意力网络的大小
        :param embed_dim: 嵌入层大小
        :param decoder_dim: 解码器 LSTM 层的维度
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

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # 注意力网络

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 词嵌入层，每个单词表示为512维向量
        self.dropout = nn.Dropout(p=self.dropout)
        # LSTMCell 输入维度为 embed_dim + encoder_dim，512+2048=2560，输出维度为 decoder_dim 512
        # Lstm的输入是词嵌入和注意力加权编码的拼接，输出是解码器的隐藏状态
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # 解码 LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # 线性层，将编码图像的特征转换为 LSTM 单元的初始隐藏状态
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # 线性层，将编码图像的特征转换为 LSTM 单元的初始细胞状态
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # 线性层，用于创建一个 sigmoid 激活的门
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # 线性层，用于找到词汇表上的分数
        self.init_weights()  # 用均匀分布初始化一些层

    def init_weights(self):
        """
        布初始化一些参数，以便更容易收敛。
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1) #将嵌入层 (self.embedding) 的权重初始化为均匀分布的随机值，范围在 -0.1 到 0.1 之间
        self.fc.bias.data.fill_(0) #将全连接层 (self.fc) 的偏置项初始化为 0
        self.fc.weight.data.uniform_(-0.1, 0.1) #将全连接层 (self.fc) 的权重初始化为均匀分布的随机值，范围在 -0.1 到 0.1 之间

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
        # 从 (batch_size, enc_image_size, enc_image_size, encoder_dim) 转换为 (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)  #每个样本的像素数量

        # 按长度递减排序输入数据；为了使用 pack_padded_sequence。下面会解释
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        
        # 对编码器输出 encoder_out) 进行重新排序。这样可以确保编码器输出与排序后的字幕长度相对应
        encoder_out = encoder_out[sort_ind]
        # 对编码字幕进行重新排序
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # 初始化 LSTM 状态
        # 此时LSTM的隐藏状态和细胞状态都是由编码图像的平均值初始化的
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # 我们不会在 <end> 位置解码，因为一旦生成 <end> 就完成了生成
        # 所以，解码长度是实际长度 - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # 创建张量以保存词预测分数和 alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device) # (batch_size, max(decode_lengths), vocab_size)，每个时间步的预测分数
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device) # (batch_size, max(decode_lengths), num_pixels)，每个时间步的注意力权重

        # 在每个时间步，通过
        # 基于解码器的上一个隐藏状态输出对编码器的输出进行注意力加权
        # 然后用上一个词和注意力加权编码在解码器中生成一个新词
        for t in range(max(decode_lengths)): # 遍历从 0 到解码长度的最大值的每个时间步 t
            batch_size_t = sum([l > t for l in decode_lengths]) # 计算当前时间步 t 的有效批次大小（即在当前时间步 t 仍有要解码的样本数量）
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # 门控标量, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            # LSTM输入是词嵌入和注意力加权编码的拼接，输出是解码器的隐藏状态h，h维度为512
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1), # 将当前时间步的嵌入词和注意力加权编码拼接在一起
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # 通过全连接层计算预测结果 (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds # 将预测结果保存到预测张量中
            alphas[:batch_size_t, t, :] = alpha # 将注意力权重保存到注意力张量中

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
