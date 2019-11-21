# GraphNeuralNetwork

## The Tools of the GraphNeuralNetwork

名称  | 类型  | 适用场景 | Github
 ---- | ----- | ------  | ------ 
 OpenNE	| 图表示学习	| 图节点表示学习，预训练 |	https://github.com/thunlp/OpenNE
Graph_nets |	图神经网络	| 基于关系模糊的图数据推理	| https://github.com/deepmind/graph_nets
DGL	| 图神经网络 |	建立图数据（可以无需通过networkx）并加载常用图神经网络 |	https://github.com/jermainewang/dgl
GPF	| 训练流程	| 基于关系数据的数据预测（节点分类、关系预测）|	https://github.com/xchadesi/GPF
networkx	| 图数据预处理	| 非大规模图数据预处理	| https://github.com/networkx/networkx
Euler	|工业级图深度学习框架	| 工业级图数据的用户研究快速进行算法创新与定制 |	https://github.com/alibaba/euler
PyG | 几何深度学习 | 适合于图、点云、流形数据的深度学习，速度比DGL快 | https://github.com/rusty1s/pytorch_geometric
PBG | 图表示学习 | 高速大规模图嵌入工具,分布式图表示学习，使用pytorch | https://github.com/facebookresearch/PyTorch-BigGraph
AliGraph | 图神经网络 | 阿里自研，大规模图神经网络平台 | https://arxiv.org/pdf/1902.08730.pdf
NSL | 图神经网络 | 主要用来训练图神经网络 | https://www.tensorflow.org/neural_structured_learning/
NGra | 图神经网络 | 支持图神经网络的并行处理框架 | https://arxiv.org/pdf/1810.08403.pdf

# The Method of Build GNN Model


关注点	| 类别	| 典型模型	| 引用 |	Github
---- | ----- | ------ | ------ | ------
**图类型**	|无向	|GNN	|  | 	
**图类型**	|	有向 |	ADGPM	|《Rethinking knowledge graph propagation for zero-shot learning》	| https://github.com/cyvius96/adgpm
**图类型**	|	异构图 |	GraphInception |	《Deep collective classification in heterogeneous information networks》 |	https://github.com/zyz282994112/GraphInception
**图类型**	|	带有边信息的图 |	G2S	|《 Graph-to-sequence learning using gated graph neural networks》 |	https://github.com/beckdaniel/acl2018_graph2seq
**图类型**	|	带有边信息的图 |		RGCN |《Modeling relational data with graph convolutionalnetworks》 | https://github.com/MichSchli/RelationPrediction /  https://github.com/masakicktashiro/rgcn_pytorch_implementation                         **聚合更新** |	谱方法卷积聚合 |	GCN		
**聚合更新** |	谱方法卷积聚合 |	ChebNet		
**聚合更新** |	非谱方法卷积聚合 |	MoNet		
**聚合更新** |	非谱方法卷积聚合 | DCNN	|《Diffusion-ConvolutionalNeural Networks》|	https://github.com/jcatw/dcnn
**聚合更新** |	非谱方法卷积聚合 |  GraphSAGE	|《GraphSage: Representation Learning on Large Graphs》| https://github.com/williamleif/GraphSAGE / https://github.com/williamleif/graphsage-simple
**聚合更新** |	注意力机制聚合	| GAT	|《Graph attention networks》|	https://github.com/PetarV-/GAT
**聚合更新** |	门更新机制	| GRU	| 《Gated graphsequence neural networks》| https://github.com/JamesChuanggg/ggnn.pytorch / https://github.com/yujiali/ggnn
**聚合更新** |	门更新机制	| LSTM	| 《Improved semanticrepresentations from tree-structured long short-term memory networks》 |	https://github.com/ttpro1995/TreeLSTMSentiment
**聚合更新** |	跳跃式连接 |	Highway GNN	| 《 Semi-supervised user geolocation via graph convolutional networks》|	https://github.com/afshinrahimi/geographconv
**聚合更新** |	跳跃式连接 |Jump Knowledge Network |	《Representation learning on graphs with jumping knowledge networks》	
**训练方法**	| 接受域的控制 |			
**训练方法**	|	采样方法	| FastGCN	|《FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling》 |	https://github.com/matenure/FastGCN
**训练方法**	|	梯度提升方法 |	Co-Training GCN		
**训练方法**	|	梯度提升方法 | Self-training GCN		
**通用框架**	| 信息传播 |	MPNN	|《Neural message passing for quantum chemistry》 |	https://github.com/brain-research/mpnn
**通用框架**	| 非局部神经网络 |	NLNN	|《 Non-local neuralnetworks》|	https://github.com/nnUyi/Non-Local_Nets-Tensorflow / https://github.com/search?q=Non-local+neural+networks
**通用框架**	| 图神经网络	|GN	| 《Relational inductive biases, deep learning, andgraph networks》 |	https://github.com/deepmind/graph_nets
