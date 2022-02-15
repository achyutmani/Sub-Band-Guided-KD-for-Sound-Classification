# Sub-Band-Guided-KD-for-Sound-Classification
This work presentes training of a lightweight student model using multiple teacher models and knowledge distillation technique. The teacher models takes full-band and sub-band spectrogrms while learning of the teacher models. Further logits of these teacher models are aggregated with three different ensemble technique and KLD loss is compued between logits of a student model and ensemble logits of teacher models. An ablation study is also conducted to supervise learning of the student model using the proposed KD technique in presence of three state-of-the-art attention mechanisms. <br>
The paper is under Major Revision at Journal of Applied Acoustics, Eslevier 
