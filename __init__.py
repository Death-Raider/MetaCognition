def examples():
    # create the dataset by running the create_ds.py script
    import semi_automated_dataset_creation.create_ds as CreateDS

    CreateDS.run(
        dataset="prhegde/preference-data-math-stack-exchange",
        section="train",
        limit=2500,
        max_workers=4
    )

    # import the ablation and sequential runners
    import Stage_5_HiPO_1Pass.ablation_runner as ablation
    import Stage_5_HiPO_1Pass.sequential_runner as sequential_run

    # import and run benchmarks
    import Experimental.eval_models as eval_models
    