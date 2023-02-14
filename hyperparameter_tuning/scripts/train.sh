python train.py datamodule.num_workers=8 \
    model.net.model_name=efficientnet_b3a model.optimizer._target_=torch.optim.Adam \
    model.optimizer.lr=0.0052086233463775385 model.optimizer.weight_decay=0 \
    datamodule.batch_size=32 trainer.max_epochs=12
