from evaluation.attribute_results_visualise import load_allfiles, build_attribute_level_metrics, build_category_level_metrics, build_weighted_avg_per_attribute, build_macro_avg_per_category,build_pipeline_overall_summary, get_master_accuracy, visualise_attribute_type


BASE_PATH = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/results_tagging"
MODELS = ["CLIP", "SigLip"]
PIPELINES=["A", "B", "C"]
METHODS = ["Direct"]

def prepaer_paths():

    paths = [{
        "model": model,
        "Pipeline": f"Pipeline {pipeline}",
        "summary_file_path": f"{BASE_PATH}/Pipeline_{pipeline}/{model}/{method}/segmented_{pipeline.lower()}_summary_metrics.csv",
        "per_class_path": f"{BASE_PATH}/Pipeline_{pipeline}/{model}/{method}/segmented_{pipeline.lower()}_metrics_per_class.csv",
        "master_path":  f"{BASE_PATH}/Pipeline_{pipeline}/{model}/{method}/segmented_{pipeline.lower()}_master.csv"
    }
    for pipeline in PIPELINES
    for model in MODELS
    for method in METHODS
    ]
    return paths


if __name__ == "__main__":
    paths = prepaer_paths()
    summary_df, class_df, master_df = load_allfiles(paths)
    attributes_to_drop = ["predicted_base_style", "predicted_colour"]
    summary_df = summary_df[~summary_df["attribute"].isin(attributes_to_drop)]

    main_table = build_category_level_metrics(class_df)
    table_path = f"{BASE_PATH}/Final_category_level_average.csv"
    main_table.to_csv(table_path)

    # main_table = build_attribute_level_metrics(summary_df)
    # table_path = f"{BASE_PATH}/'Final_attribute_level_average.csv'"
    # main_table.to_csv(table_path)

    main_table = build_weighted_avg_per_attribute(summary_df)
    table_path = f"{BASE_PATH}/Final_weighted_attribute_level_average.csv"
    main_table.to_csv(table_path)
    visualise_attribute_type(f"{BASE_PATH}/Final_weighted_attribute_level_average.csv", BASE_PATH)

    main_table = build_macro_avg_per_category(summary_df)
    table_path = f"{BASE_PATH}/Final_weighted_category_level_average.csv"
    main_table.to_csv(table_path)

    main_table = build_pipeline_overall_summary(summary_df).T
    table_path = f"{BASE_PATH}/Final_weighted_overall.csv"
    main_table.to_csv(table_path)

    main_table = get_master_accuracy(master_df)
    table_path = f"{BASE_PATH}/overall_accuracy.csv"
    main_table.to_csv(table_path)