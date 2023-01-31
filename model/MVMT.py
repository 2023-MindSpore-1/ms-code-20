import mindspore 
import mindspore.nn as mindnn
from mindspore import nn
import mindspore.ops as ops
import numpy as np
from mindspore import Parameter, Tensor

#Channel-Wise CNN Block for Local Spatial Feature Learning

class ChannelLayer(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(ChannelLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.SequentialCell([
            nn.Dense(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dense(channel // reduction, channel, bias=False),
            nn.Sigmoid()]
        )

    def construct(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelBlock(nn.Cell):
    def __init__(self, in_features):
        super(ChannelBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      ]

        self.se = ChannelLayer(in_features)
        self.conv_block = nn.SequentialCell(conv_block)


    def forward(self, x):
        out = self.conv_block(x)
        out = self.se(out)
        return x + out


# Multi-View Self-Attention Block for Global Semantic Feature Learning

def attention(q, k, v, d_k):
    scores = nn.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    softmax = ops.Softmax()
    scores = softmax(scores)
    output = nn.matmul(scores, v)

    return output

class Muview_Attention(nn.Cell):
    # multiheadattention (batchsize,graph,node_num,hidden_size)
    def __init__(self,GCN_size,d_model):
        super().__init__()

        self.d_model = d_model
        self.q_linear = nn.Dense(GCN_size, d_model)
        self.k_linear = nn.Dense(GCN_size, d_model)
        self.S_linear=  nn.Dense(d_model, d_model)
        self.a1=Parameter(Tensor(np.ones((1))), name="w_1",requires_grad=True)
        self.a2=Parameter(Tensor(np.ones((1))), name="w_2",requires_grad=True)
        self.a3=Parameter(Tensor(np.ones((1))), name="w_3",requires_grad=True)


    def construct(self, risk_in,road_in,poi_in):
        output=[]
        bs,num,D=road_in.shape
        concat_op = ops.Concat()
        reshape = ops.Reshape()
        v=reshape(concat_op([risk_in,road_in,poi_in]),(3,-1,num,D))
        for i in range(3):
            k = self.k_linear(v)
            q = self.q_linear(v)
            outs= attention(q, k, v, self.d_model)
            output.append(outs)

        risk_out=((output[0])[0,:,:,:]+(output[1])[0,:,:,:]+(output[2])[0,:,:,:])/3
        road_out=((output[0])[1,:,:,:]+(output[1])[1,:,:,:]+(output[2])[1,:,:,:])/3
        poi_out=((output[0])[2,:,:,:]+(output[1])[2,:,:,:]+(output[2])[2,:,:,:])/3

        risk=self.a1[0]*(risk_out)+(1-self.a1[0])*risk_in
        road=self.a2[0]*(road_out)+(1-self.a2[0])*road_in
        poi=self.a3[0]*(poi_out)+(1-self.a3[0])*poi_in
        last_out=self.S_linear(risk+road+poi)
        return last_out



class Muti_GCN(nn.Cell):
    def __init__(self,num_of_graph_feature,nums_of_graph_filters):

        super(Muti_GCN,self).__init__()
        self.road_gcn = nn.CellList()
        self.mymodel=Muview_Attention(64,64)

        for idx,num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.road_gcn.append(GCN_Layer(num_of_graph_feature,num_of_filter))
            else:
                self.road_gcn.append(GCN_Layer(nums_of_graph_filters[idx-1],num_of_filter))

        self.risk_gcn = nn.ModuleList()
        for idx,num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.risk_gcn.append(GCN_Layer(num_of_graph_feature,num_of_filter))
            else:
                self.risk_gcn.append(GCN_Layer(nums_of_graph_filters[idx-1],num_of_filter))

        self.poi_gcn = nn.ModuleList()
        for idx,num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.poi_gcn.append(GCN_Layer(num_of_graph_feature,num_of_filter))
            else:
                self.poi_gcn.append(GCN_Layer(nums_of_graph_filters[idx-1],num_of_filter))
        self.mymodel=Muview_Attention(64,64)


    def forward(self,graph_feature,road_adj,risk_adj,poi_adj):

        batch_size,T,D1,N = graph_feature.shape

        road_graph_output = graph_feature.view(-1,D1,N).permute(0,2,1).contiguous()
        for gcn_layer in self.road_gcn:
            road_graph_output = gcn_layer(road_graph_output,road_adj)


        risk_graph_output = graph_feature.view(-1,D1,N).permute(0,2,1).contiguous()
        for gcn_layer in self.risk_gcn:
            risk_graph_output = gcn_layer(risk_graph_output,risk_adj)


        if poi_adj is not None:
            poi_graph_output = graph_feature.view(-1,D1,N).permute(0,2,1).contiguous()
            for gcn_layer in self.poi_gcn:
                poi_graph_output = gcn_layer(poi_graph_output,poi_adj)

        graph_output=self.mymodel(risk_graph_output,road_graph_output,poi_graph_output)
  
        graph_output=graph_output.reshape(batch_size,T,-1,N)


        return graph_output





class STGeoModule2(nn.Cell):
    def __init__(self,grid_in_channel,num_of_lstm_layers,seq_len,
                lstm_hidden_size,num_of_target_time_feature):
        """[summary]

        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_lstm_layers {int} -- the number of LSTM layers
            seq_len {int} -- the time length of input
            lstm_hidden_size {int} -- the hidden size of LSTM
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
        """
        super(STGeoModule2,self).__init__()
        self.grid_lstm = nn.LSTM(grid_in_channel,lstm_hidden_size,num_of_lstm_layers,batch_first=True)
        self.grid_att_fc1 = nn.Dense(in_features=lstm_hidden_size,out_features=1)
        self.grid_att_fc2 = nn.Dense(in_features=num_of_target_time_feature,out_features=seq_len)
        self.grid_att_bias = Parameter(mindspore.ops.Zeros(1))
        self.grid_att_softmax = nn.Softmax(dim=-1)
        self.SE=SEBlock(grid_in_channel)


    def forward(self,grid_input,target_time_feature):
        batch_size,T,D,W,H = grid_input.shape
        grid_input = grid_input.reshape(-1,D,H,W)
        conv_output=self.SE(grid_input)

        conv_output = conv_output.view(batch_size,-1,D,W,H)\
                        .permute(0,3,4,1,2)\
                        .contiguous()\
                        .view(-1,T,D)
        lstm_output,_ = self.grid_lstm(conv_output)

        grid_target_time = mindspore.ops.expand_dims(target_time_feature,1).repeat(1,W*H,1).view(batch_size*W*H,-1)
        grid_att_fc1_output = mindspore.ops.Squeeze(self.grid_att_fc1(lstm_output))
        grid_att_fc2_output = self.grid_att_fc2(grid_target_time)
        grid_att_score = self.grid_att_softmax(F.relu(grid_att_fc1_output+grid_att_fc2_output+self.grid_att_bias))
        grid_att_score = grid_att_score.view(batch_size*W*H,-1,1)
        grid_output = mindspore.ops.ReduceSum(lstm_output * grid_att_score,dim=1)
        grid_output = grid_output.view(batch_size,-1,W,H).contiguous()
        return grid_output





class MVMT(nn.Cell):
    def __init__(self,grid_in_channel,num_of_lstm_layers,seq_len,pre_len,
                lstm_hidden_size,num_of_target_time_feature,
                num_of_graph_feature,nums_of_graph_filters,
                north_south_map,west_east_map):

        super(MVMT,self).__init__()
        self.st_geo_module = STGeoModule2(grid_in_channel,3,seq_len,
                                          64,num_of_target_time_feature)
        self.st_geo_module2 = STGeoModule2(grid_in_channel,num_of_lstm_layers,seq_len,
                                        lstm_hidden_size,num_of_target_time_feature)
        self.Muti_GCN_f=Muti_GCN(num_of_graph_feature,nums_of_graph_filters)
        self.Muti_GCN_c=Muti_GCN(num_of_graph_feature,nums_of_graph_filters)
        self.time_output_f=Time_pro(seq_len,num_of_lstm_layers,lstm_hidden_size,num_of_target_time_feature)
        self.time_output_c=Time_pro(seq_len,3,64,num_of_target_time_feature)
        self.north_south_map=north_south_map
        self.west_east_map=west_east_map

        fusion_channel = 16
        self.grid_weigth_f = nn.Conv2d(in_channels=lstm_hidden_size,out_channels=fusion_channel,kernel_size=1)
        self.grid_weigth_c = nn.Conv2d(in_channels=64,out_channels=fusion_channel,kernel_size=1)
        self.graph_weigth_f = nn.Conv2d(in_channels=lstm_hidden_size,out_channels=fusion_channel,kernel_size=1)
        self.graph_weigth_c = nn.Conv2d(in_channels=64,out_channels=fusion_channel,kernel_size=1)
        # nyc
        f_num=243
        c_num=75
        batch_size=32
        self.output_layer1_f = nn.Dense(fusion_channel*north_south_map*west_east_map,pre_len*north_south_map*west_east_map)
        self.output_layer1_c = nn.Dense(fusion_channel*int((north_south_map/2)*(west_east_map/2)),pre_len*int((north_south_map/2)*(west_east_map/2)))


    def forward(self,f_train_feature,c_train_feature,target_time_feature,f_graph_feature,c_graph_feature,
                f_road_adj,c_road_adj,f_risk_adj,c_risk_adj,f_poi_adj,c_poi_adj,grid_node_map_f,grid_node_map_c,trans):

# grid_output
        batch_size,_,_,_,_=c_train_feature.shape
        f_grid_output = self.st_geo_module2(f_train_feature,target_time_feature)
# grid_output
        c_grid_output = self.st_geo_module(c_train_feature,target_time_feature)


# graph_output:
        c_graph_output=self.Muti_GCN_c(c_graph_feature,c_road_adj,c_risk_adj,c_poi_adj)
        f_graph_output=self.Muti_GCN_f(f_graph_feature,f_road_adj,f_risk_adj,f_poi_adj)


# # # coarse to finer
        batch_size1,T,_,c_N=c_graph_output.shape
        batch_size,T,_,f_N=f_graph_output.shape
        c_graph_output=c_graph_output.reshape(batch_size1*T,-1,c_N)
        cf_out=mindspore.ops.ReLU(mindspore.ops.MatMul(c_graph_output,trans))
        f1_graph_output=f_graph_output+0.2*cf_out.reshape(batch_size1,T,-1,f_N)

# finer to coarse
        f_graph_output=f_graph_output.reshape(batch_size*T,-1,f_N)
        trans2=trans.permute(0,2,1)
        fc_out=mindspore.ops.ReLU(mindspore.ops.MatMul(f_graph_output,trans2))
        c_graph_output=c_graph_output.reshape(batch_size1,T,-1,c_N)
        c1_graph_output=c_graph_output+0.8*fc_out.reshape((batch_size,T,-1,c_N))
        graph_output_c1=self.time_output_c(c1_graph_output,target_time_feature)
        graph_output_f1=self.time_output_f(f1_graph_output,target_time_feature)
        graph_output_c=graph_output_c1
        graph_output_f=graph_output_f1
        graph_output_f=graph_output_f.permute(0,2,1)
        batch_size,_,_=graph_output_f.shape
        grid_node_map_tmp_f = mindspore.Tensor.from_numpy(grid_node_map_f)\
                            .to(graph_output_f.device)\
                            .repeat(batch_size,1,1)
        graph_output_f = mindspore.ops.BatchMatMul(grid_node_map_tmp_f,graph_output_f)\
                            .permute(0,2,1)\
                            .view(batch_size,-1,self.north_south_map,self.west_east_map)
        graph_output_c=graph_output_c.permute(0,2,1)
        batch_size,_,_=graph_output_c.shape

        grid_node_map_tmp_c =mindspore.Tensor.from_numpy(grid_node_map_c)\
                            .to(graph_output_f.device)\
                            .repeat(batch_size,1,1)

        graph_output_c = mindspore.ops.BatchMatMul(grid_node_map_tmp_c,graph_output_c)\
                            .permute(0,2,1)\
                            .view(batch_size,-1,int((self.north_south_map)/2),int((self.west_east_map)/2))



        f_grid_output = self.grid_weigth_f(f_grid_output)
        graph_output_f = self.graph_weigth_f(graph_output_f)
        f_fusion_output = (f_grid_output + graph_output_f).view(batch_size,-1)
        f_final_output = self.output_layer1_f(f_fusion_output).view(batch_size,-1,self.north_south_map,self.west_east_map)
        c_grid_output = self.grid_weigth_c(c_grid_output)
        graph_output_c = self.graph_weigth_c(graph_output_c)
        c_fusion_output = (c_grid_output + graph_output_c).view(batch_size,-1)
        c_final_output = self.output_layer1_c(c_fusion_output).view(batch_size,-1,int(self.north_south_map/2),int(self.west_east_map/2))

        return f_final_output,c_final_output
