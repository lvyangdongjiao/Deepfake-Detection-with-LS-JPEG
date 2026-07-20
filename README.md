目的：针对Deepfake伪造人脸检测算法在JPEG压缩场景下性能显著退化的问题，本文基于Xception网络，提出并验证了一种结合标签平滑（Label Smoothing, LS）与JPEG增强的训练优化策略，旨在提升模型在真实部署环境下的压缩鲁棒性。
<img width="356" height="291" alt="3e6b3eea078dbe6e3ba49c82d77ceb34" src="https://github.com/user-attachments/assets/afcf457e-d74c-4da4-a0f3-f82bff24e1aa" />

方法：实验基于FaceForensics++与DF40数据集，选取DeepFaceLab与FaceSwap两种主流伪造方法构建训练集，图像统一裁剪为299×299像素，并采用随机水平翻转进行数据增强。在测试阶段设计三阶段图像退化管道模拟实际压缩环境。优化策略包括：在训练中引入随机JPEG压缩（质量因子30–90）与标签平滑正则化，单独及组合评估其对模型抗压缩能力的影响。
<img width="438" height="117" alt="image" src="https://github.com/user-attachments/assets/52b6d642-5f48-46f9-ae09-1c97d3725609" />


结果：基线Xception模型在高质量（QF=100）下准确率为99.3%，但在强压缩（QF=30）下准确率骤降至61.3%，性能跌幅达38%。单独使用JPEG增强或标签平滑均未能显著改善，甚至出现负效应。而JPEG+LS组合策略在QF=50下准确率提升至82.17%，较基线提升20.87%；在QF=30下准确率达70.4%，分离度指标（0.5498）与5-NN准确率（87.14%）均为最优，t-SNE可视化显示真实与伪造样本形成清晰聚类边界。跨数据集（中国人脸GAN图像）泛化实验中，组合模型准确率为89.2%，验证了其在跨人种场景下的适应性。但在扩散模型（CollabDiff）检测任务中，所有模型准确率均降至30–40%，表明当前方法对非GAN生成机制缺乏泛化能力。

<img width="455" height="261" alt="image" src="https://github.com/user-attachments/assets/b5acf231-9138-4e0f-b693-824036ba85a4" />
<img width="409" height="160" alt="image" src="https://github.com/user-attachments/assets/fbf58ebb-bfbe-42c0-8900-980d4a13b13b" />

结论：JPEG增强与标签平滑的协同策略能有效缓解压缩带来的域偏移问题，显著提升Xception模型在实际压缩环境下的检测鲁棒性。然而，模型仍局限于学习GAN特有的伪造指纹，对扩散模型等新型生成方法失效，未来需探索与算法无关的通用伪造特征表示及多模态融合策略。
