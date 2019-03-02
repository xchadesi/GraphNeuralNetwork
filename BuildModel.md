# The Method of Build GNN Model

表头  | 表头  | 表头
 
关注点	| 类别	| 典型模型	| 引用 |	Github
---- | ----- | ------ | ------ | ------
图类型	|无向	|GNN	|  | 	
图类型	|	有向 |	ADGPM	|《Rethinking knowledge graph propagation for zero-shot learning》	| https://github.com/cyvius96/adgpm
图类型	|	异构图 |	GraphInception |	《Deep collective classification in heterogeneous information networks》 |	https://github.com/zyz282994112/GraphInception
图类型	|	带有边信息的图 |	G2S	|《 Graph-to-sequence learning using gated graph neural networks》 |	https://github.com/beckdaniel/acl2018_graph2seq
图类型	|	带有边信息的图 |		RGCN |《Modeling relational data with graph convolutionalnetworks》 | https://github.com/MichSchli/RelationPrediction /  https://github.com/masakicktashiro/rgcn_pytorch_implementation                                                                                                 
聚合更新 |	谱方法卷积聚合 |	GCN		
聚合更新 |	谱方法卷积聚合 |	ChebNet		
聚合更新 |	非谱方法卷积聚合 |	MoNet		
聚合更新 |	非谱方法卷积聚合 | DCNN	|《Diffusion-ConvolutionalNeural Networks》|	https://github.com/jcatw/dcnn
聚合更新 |	非谱方法卷积聚合 |  GraphSAGE	|《GraphSage: Representation Learning on Large Graphs》| https://github.com/williamleif/GraphSAGE / https://github.com/williamleif/graphsage-simple
聚合更新 |	注意力机制聚合	| GAT	|《Graph attention networks》|	https://github.com/PetarV-/GAT
聚合更新 |	门更新机制	| GRU	| 《Gated graphsequence neural networks》| https://github.com/JamesChuanggg/ggnn.pytorch / https://github.com/yujiali/ggnn
聚合更新 |	门更新机制	| LSTM	| 《Improved semanticrepresentations from tree-structured long short-term memory networks》 |	https://github.com/ttpro1995/TreeLSTMSentiment
聚合更新 |	跳跃式连接 |	Highway GNN	| 《 Semi-supervised user geolocation via graph convolutional networks》|	https://github.com/afshinrahimi/geographconv
聚合更新 |	跳跃式连接 |Jump Knowledge Network |	《Representation learning on graphs with jumping knowledge networks》	
训练方法	| 接受域的控制 |			
训练方法	|	采样方法	| FastGCN	|《FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling》 |	https://github.com/matenure/FastGCN
训练方法	|	梯度提升方法 |	Co-Training GCN		
训练方法	|	梯度提升方法 | Self-training GCN		
通用框架	| 信息传播 |	MPNN	|《Neural message passing for quantum chemistry》 |	https://github.com/brain-research/mpnn
通用框架	| 非局部神经网络 |	NLNN	|《 Non-local neuralnetworks》|	https://github.com/nnUyi/Non-Local_Nets-Tensorflow / https://github.com/search?q=Non-local+neural+networks
通用框架	| 图神经网络	|GN	| 《Relational inductive biases, deep learning, andgraph networks》 |	https://github.com/deepmind/graph_nets
