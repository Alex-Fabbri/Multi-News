"""Define RNN-based encoders.
(Irene) This is modified by me, adding another feature (src_sents)



"""
from __future__ import division
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

from torch.nn.utils.rnn import pad_sequence


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers, # this is 1.
                        dropout=dropout,
                        bidirectional=bidirectional)
        # self.rnn is a LSTM(128, 256, bidirectional=True)  # input dim; output dim;


        # init another sentence-level layer: this is shared by all the sentences, why 4?
        self.sent_rnn = nn.LSTM(4*embeddings.embedding_size,hidden_size, num_layers= 1, dropout=dropout, bidirectional=True)
        # (512,256)
        # import pdb;pdb.set_trace()


        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)


    def build_sentence_layer(self,memory_bank,src_sents):
        '''
        In this method we define sentence level representation. (This is the old version)
        :param memory_bank:
        :param encoder_final:
        :param src_sents:
        :return: sentence embeddings
        '''
        # print('Memory..', memory_bank.size()) # torch.Size([200, 2, 512]) TODO: this is the output
        # #
        # print ('encoder_final..',encoder_final) #


        if isinstance(memory_bank, torch.nn.utils.rnn.PackedSequence):

            memory_bank = nn.utils.rnn.pad_packed_sequence(memory_bank)[0] # as after unpack it is a tuple

        hidden = memory_bank.permute(1,0,2) # size: (2,200,512)

        # print ('in func...', src_sents)

        # in each case for the current batch, send the last hidden output as the input to the sent_lstm layer
        batch_input_list = []
        for output,sent_id in zip(hidden,src_sents): # so we have batch_size to be 1

            common_len = len(sent_id)

            output = output.unsqueeze(1)
            sent_input_list = []

            # firs id
            start_ind_sent_id = 0
            start_ind = sent_id[start_ind_sent_id]


            while (start_ind < output.size()[0]) and (start_ind_sent_id < sent_id.size()[0]):


                # add
                sent_input_list.append(output[start_ind])

                # both ids move to the next
                start_ind_sent_id += 1
                if start_ind_sent_id < sent_id.size()[0]:
                    start_ind += sent_id[start_ind_sent_id]
                else:
                    break


            # FEB 10, len check
            if len(sent_input_list) < common_len:
                # pad with zero
                pad_size = output[0].size()
                zeros = torch.zeros(pad_size, dtype=torch.float32).cuda()
                pad_list = [zeros]* (common_len-len(sent_input_list))

                sent_input_list = sent_input_list + pad_list


            sent_input = torch.cat(sent_input_list,0).unsqueeze(1) # (n_sent, batch_size=1,dim=512)


            batch_input_list.append(sent_input)



        # print ([x.size() for x in batch_input_list])
        # [torch.Size([18, 1, 512]), torch.Size([15, 1, 512]), torch.Size([18, 1, 512]), torch.Size([18, 1, 512]), torch.Size([18, 1, 512])]


        batch_input_list_concat = torch.cat(batch_input_list,1)

        # get the id of sent length:
        sent_output, (h_, c_) = self.sent_rnn(batch_input_list_concat)
        # LSTM(512, 256, bidirectional=True), sent_output has the same shape with batch_input_list_concat


        #sent_output: shape(number of sents or step, batch_size, dim) (9, 2, 512), number of sents or step can be different
        # print ('Encoder Sentence_output...',sent_output.size())
        return sent_output


    def sent_level_encoder(self, memory_bank):
        '''
        This is the sentence level encoder, it takes a bunch of sentence encoding,
        then feed into another sentence level rnn
        :param memory_bank: sentence encoding ( a list of packed)
        :return: output of the rnn layer
        '''

        if isinstance(memory_bank, torch.nn.utils.rnn.PackedSequence):
            memory_bank_unpacked = nn.utils.rnn.pad_packed_sequence(memory_bank)[0].permute(1,0,2)# as after unpack it is a tuple
        # memory_bank_unpacked size: torch.Size([42, 9, 512]) # [seq_len,batch_size,512]


        # take the last hiddent state of each
        last_hidden = [x[-1].unsqueeze(0) for x in memory_bank_unpacked]
        last_hidden = torch.cat(last_hidden, 0).unsqueeze(0) # size is [1,9,512]

        sent_output, (h_, c_) = self.sent_rnn(last_hidden)


        return sent_output

    def forward_new(self, src, src_sents=None, lengths=None):
        "New Forward`"
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        s_len, batch, emb_dim = emb.size() # (185 16 128), s_len is sequence_len.

        # 2333 TODO: change starts here


        # Feb15: we break this into sentences

        # iterate each batch..
        input_embeddings=emb.permute(1,0,2)

        final_memory_bank = []
        final_encoder_final = []

        final_sent_output = []

        for batch_id in range(batch):

            # this is the input to word-level lstm
            current_sequence = input_embeddings[batch_id] # size id (sequence_len, emb_dim)
            # break this into multiple sentences according to the sentence lengths, and input to the rnn
            # sent len check, define len_sequence to be: tensor([26, 17, 21, 23, 19, 26, 10, 42], device='cuda:0')
            if torch.sum(src_sents[batch_id]) >= s_len:
                # if exceeds the total length, then their is a bug
                len_sequence = src_sents[batch_id][:-1]
            else:
                len_sequence = src_sents[batch_id]

            counter = 0


            feeding_as_a_batch = []
            lengths_as_a_batch = []
            lengths = []
            actual_len = 0
            for idx in len_sequence:
                if (counter < s_len ) and (idx != 0):
                    actual_len += 1
                    # from the current_sequence, add to the rnn
                    feeding_sequence = current_sequence[counter:counter+idx].unsqueeze(0)
                    feeding_as_a_batch.append(feeding_sequence.permute(1,0,2)) #feeding_sequence size = [1,26,128]
                    counter += idx
                    # feed into rnn
                lengths_as_a_batch.append(idx)

            feeding_as_a_batch_padded = torch.cat([x for x in pad_sequence(feeding_as_a_batch,batch_first=True)],1)
            # feed into rnn size: torch.Size([42, 9, 128]) -> [max, batch_size, dim]
            max_dim = feeding_as_a_batch_padded.size()[0]
            lengths_as_a_batch = [max_dim for x in range(actual_len)]
            # lengths_as_a_batch = [item for sublist in lengths_as_a_batch for item in sublist]

            if lengths_as_a_batch is not None and not self.no_pack_padded_seq:
                # Lengths data is wrapped inside a Tensor.
                packed_emb_rnn_input = pack(feeding_as_a_batch_padded, lengths_as_a_batch)


            # feed into!
            memory_bank, encoder_final = self.rnn(packed_emb_rnn_input)

            # feed into sentence_level
            sent_output = self.sent_level_encoder(memory_bank)
            final_sent_output.append(sent_output.view(-1,4*emb_dim))


            if lengths is not None and not self.no_pack_padded_seq:
                memory_bank = unpack(memory_bank)[0]


            # we need to get the original output, before padded
            revised_memory_bank = memory_bank.permute(1, 0, 2)
            memory_bank_unpadded_list = []



            for idx in range(actual_len):
                memory_bank_unpadded_list.append(revised_memory_bank[idx][:len_sequence[idx]])
            unpadded_memory_bank = torch.cat(memory_bank_unpadded_list,0) # size is [sequence_len,512] # need to pad or truncate
            actual_size = unpadded_memory_bank.size()[0]
            if actual_size >= s_len:
                padded_memory_bank = unpadded_memory_bank[:s_len]
                # print ('Size is okk..', padded_memory_bank.size())
            else:
                # pad with zero
                pad_size = s_len - actual_size
                padded_memory_bank = F.pad(unpadded_memory_bank, (0,0,0,pad_size), 'constant',0.0)
                # print ('Padded...',unpadded_memory_bank.size(),pad_size,padded_memory_bank.size())
            # print (actual_size,s_len,padded_memory_bank.size())
            final_memory_bank.append(padded_memory_bank.unsqueeze(1))
            # finish processing on memory bank


            if self.use_bridge:
                encoder_final = self._bridge(encoder_final)

            final_encoder_final.append(tuple([x[:,-1,:].unsqueeze(1) for x in encoder_final]))

        # add unpacked from final_memory_bank
        final_memory_bank = torch.cat(final_memory_bank,1) # [200, 2, 512], ready to return


        # join the encoder_final
        hs = []
        cs = []
        for (h,c) in final_encoder_final:
            hs.append(h)
            cs.append(c)
        hs = torch.cat(hs,1)
        cs = torch.cat(cs,1)
        # encoder_final
        final_encoder_final = tuple([hs,cs]) # ready to return

        # sent output
        final_sent_output = pad_sequence(final_sent_output)  # size [9,2,512], ready to return

        # 2333 TODO: change finish here

        # import pdb;pdb.set_trace()

        return final_encoder_final, final_memory_bank, final_sent_output

    def forward(self, src, src_sents=None, lengths=None):
        "forward_original"
        #print ('Original!')
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        s_len, batch, emb_dim = emb.size() # (185 16 128), s_len is changeable.


        packed_emb = emb


        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)



        memory_bank, encoder_final = self.rnn(packed_emb) # output, (hidden, cell), unpack using pad_packed_sequence(), encoder_final is the last state, a list (contains the batch)
        # encoder_final size: a list, len is the batch size; for each item, size [2, 2, 256]


        # memory_bank is the output
        # output, (hidden, cell), unpack using pad_packed_sequence()
        # self.rnn is a LSTM(128, 256, bidirectional=True) # input dim; output dim;

        # print ('forwarding... src_sents',src_sents)

        # get sentence embedding
        sent_output = self.build_sentence_layer(memory_bank,src_sents)
        # sent_output size: torch.Size([9, 2, 512])
        # print ('We need...!!!',src_sents.size(),src_sents)


        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
        # memory_bank size torch.Size([200, 2, 512])


        # encoder_final: a tuple of 2 (batch size)
        # each of it has the size of torch.Size([2, 2, 256])

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        # encoder_final same shape as before

        return encoder_final, memory_bank, sent_output

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs
