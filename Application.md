# The Application of the GraphNeuralNetwork

领域  | 应用  | 算法 | 引用 | Github
 ---- | ----- | ------ |  ----- | ------
通用  |关系预测 |RGCN| 《Modeling Relational Data with Graph Convolutional Networks》 | [rgcn_pytorch_implementation](https://github.com/masakicktashiro/rgcn_pytorch_implementation) 
 通用 | 关系预测 | SEAL| 《Link Prediction Based on Graph Neural Networks》 | [SEAL](https://github.com/muhanzhang/SEAL)
通用 |节点分类  |  |   |  
通用 |社区检测  |  |《Improved Community Detection using Deep Embeddings from Multilayer Graphs》   | 
通用 |社区检测  | Hierarchical GNN |  《Supervised Community Detection with Hierarchical Graph Neural Networks》 | 
通用 | 图分类 |  | 《Graph Classification using Structural Attention》  | 
通用 | 图分类 |DGCNN  |《An End-to-End Deep Learning Architecture for Graph Classification》| [pytorch_DGCNN](https://github.com/muhanzhang/pytorch_DGCNN)
通用 | 推荐 |  GCN|  《Graph Convolutional Neural Networks for Web-Scale Recommender Systems》 | 
通用|图生成  | NetGAN  | 《 Net-gan: Generating graphs via random walks》  | 
通用| 图生成 | GraphRNN |  《GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models》 | 
通用 | 图生成 |MolGAN  | 《 Molgan: An implicit generative model for small molecular graphs》  | 
决策优化 | 旅行商问题 | GNN |  《Learning to Solve NP-Complete Problems: A Graph Neural Network for Decision TSP》《Attention solves your tsp》 | [attention-tsp](https://github.com/machine-reasoning-ufrgs/TSP-GNN https://github.com/wouterkool/attention-tsp)
决策优化 | 规划器调度 | GNN |  《Adaptive Planner Scheduling with Graph Neural Networks》《Revised note on learning quadratic assignment with graph neural networks》 | 
 决策优化| 组合优化 | GCN structure2vec |  《Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search》《 Learning combinatorial optimization algorithms over graphs》 | [NPHard](https://github.com/IntelVCL/NPHard)
交通 | 出租车需求预测 |  |  《Deep Multi-View Spatial-Temporal Network for Taxi Demand Prediction》 | [DMVST-Net](https://github.com/huaxiuyao/DMVST-Net)
交通 | 交通流量预测 |  | 《Spatio-Temporal Graph Convolutional Networks:A Deep Learning Framework for Traffic Forecasting》  |[STGCN-PyTorch](https://github.com/FelixOpolka/STGCN-PyTorch)
交通 | 交通流量预测 |  |  《DIFFUSION CONVOLUTIONAL RECURRENT NEURAL NETWORK: DATA-DRIVEN TRAFFIC FORECASTING》 | [DCRNN](https://github.com/liyaguang/DCRNN)
传感网络 | 传感器布局 |  |   《Distributed Graph Layout for Sensor Networks》| 
区域安全 | 疾病传播 |  |  《Predicting and controlling infectious disease epidemics using temporal networks》 | 
区域安全 | 城市人流预测 |  |  《FCCF: Forecasting Citywide Crowd Flows Based on Big Data》 | 
社交网络 | 影响力预测 | GCN/GAT | 《DeepInf: Social Influence Prediction with Deep Learning》  | [DeepInf](https://github.com/xptree/DeepInf)
社交网络 | 转发动作预测 |  | 《Social Influence Locality for Modeling Retweeting Behaviors》  | 
社交网络 | 转发动作预测 |  | 《 Predicting Retweet via Social Influence Locality》  | 
 文本|	文本分类|	GCN	|"《Diffusion-convolutional neural networks》《 Convolutionalneural networks on graphs with fast localized spectral filtering》《Knowledgetransfer for out-of-knowledge-base entities : A graph neuralnetwork approach》《 Deep convolutional networks
on graph-structured data》《 Semi-supervised classification with graph convolutional networks》《 Geometric deep learning on graphs and manifolds using mixture model cnns》"|[dcnn-tensorflow](https://github.com/RicardoZiTseng/dcnn-tensorflow)
		GAT	《Graph attention networks》	
		DGCNN	《Large-scale hierarchical text classification with recursively regularized deep graph-cnn》	https://github.com/HKUST-KnowComp/DeepGraphCNNforTexts
		Text GCN	"《Graph convolutional networks for
text classification》"	https://github.com/yao8839836/text_gcn
		Sentence LSTM	"《 Sentence-state LSTM for text
representation》"	https://github.com/leuchine/S-LSTM
	序列标注(POS, NER)	Sentence LSTM	"《 Sentence-state LSTM for text
representation》"	https://github.com/leuchine/S-LSTM
	语义分类	Tree LSTM	《 Improved semantic representations from tree-structured long short-term memorynetworks》	https://github.com/ttpro1995/TreeLSTMSentiment
	语义角色标注	Syntactic GCN	《Encoding sentences with graph convolutional networks for semantic role labeling》	
	机器翻译	Syntactic GCN	"《Graph convolutional encoders for syntax-aware neural machine translation》
《 Exploiting semantics in neural machine translation with graph convolutional networks》"	
		GGNN	"《 Graph-to-sequence learning
using gated graph neural networks. 》"	https://github.com/beckdaniel/acl2018_graph2seq
	关系抽取	Tree LSTM	"《 End-to-end relation extraction using
lstms on sequences and tree structures》"	
		Graph LSTM	"《Crosssentence
n-ary relation extraction with graph lstms》
《 N-ary relation
extraction using graph state lstm》"	https://github.com/freesunshine0316/nary-grn
		GCN	《 Graph convolution over pruned dependency trees improves relation extraction》	https://github.com/qipeng/gcn-over-pruned-trees
	事件抽取	Syntactic GCN	"《 Jointly multiple events extraction
via attention-based graph information aggregation》
《. Graph convolutional networks
with argument-aware pooling for event detection》"	https://github.com/lx865712528/JMEE
	文本生成	Sentence LSTM	"《A graph-to-sequence
model for amr-to-text generation》"	
		GGNN	"《 Graph-to-sequence learning
using gated graph neural networks》"	
	阅读理解 	Sentence LSTM	《Exploring graph-structured passage representation for multihop reading comprehension with graph neural networks》	
图像/视频	社会关系理解	GRM	《Deep reasoning with knowledge graph for social relationship understanding》	https://github.com/wzhouxiff/SR
	图像分类	GCN	"《 Few-shot learning with graph neural
networks》
《Zero-shot recognition via semantic
embeddings and knowledge graphs》"	"https://github.com/louis2889184/gnn_few_shot_cifar100
https://github.com/JudyYe/zero-shot-gcn"
		GGNN	《 Multi-label zero-shot learning with structured knowledge graphs》	https://people.csail.mit.edu/weifang/project/vll18-mlzsl/
		ADGPM	《Rethinking knowledge graph propagation for zero-shot learning》	https://github.com/cyvius96/adgpm
		GSNN	《The more you know: Using knowledge graphs for image classification》	https://github.com/KMarino/GSNN_TMYN
	视觉问答	GGNN	"《Graph-structured
representations for visual question answering》
《Deep reasoning with knowledge graph for social relationship understanding》 "	
	领域识别	GCNN	"《Iterative visual
reasoning beyond convolutions》"	https://github.com/coderSkyChen/Iterative-Visual-Reasoning.pytorch
	语义分割	Graph LSTM	"《 Interpretable
structure-evolving lstm》
《 Semantic object
parsing with graph lstm》"	
		GGNN	"《Large-scale point cloud semantic
segmentation with superpoint graphs》"	https://github.com/loicland/superpoint_graph
		DGCNN	《Dynamic graph cnn for learning on point clouds》	https://github.com/af13s/dgcnn-amino
		3DGNN	"《 3d graph neural
networks for rgbd semantic segmentation》"	https://github.com/yanx27/3DGNN_pytorch
生物科技	物理系统	IN	《 Interaction networks for learning about objects, relations and physics》	"https://github.com/higgsfield/interaction_network_pytorch
https://github.com/jaesik817/Interaction-networks_tensorflow"
		VIN	《 Visual interaction networks: Learning a physics simulator from video》	
		GN	《 Graph networks as learnable physics engines for inference and control》	https://github.com/fxia22/gn.pytorch
	分子指纹	GCN	《Convolutional networks on graphs for learning molecular fingerprints》	https://github.com/fllinares/neural_fingerprints_tf
		NGF	"《Molecular graph convolutions: 
moving beyond fingerprints》"	
	蛋白质界面预测	GCN	"《Protein interface
prediction using graph convolutional networks》"	https://github.com/fouticus/pipgcn
	药物副作用预测	Decagon	"《Modeling polypharmacy
side effects with graph convolutional networks》"	https://github.com/miliana/DecagonPython3
	疾病分类	PPIN	"《Hybrid approach of relation network
and localized graph convolutional filtering for breast cancer subtype classification》"	
|  |  |   | 
 |  |  |   | 
 |  |  |   | 
 |  |  |   | 
 |  |  |   | 
 |  |  |   | 
 
