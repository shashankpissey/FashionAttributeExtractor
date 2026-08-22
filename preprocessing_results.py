from evaluation.preprocessing_results_visualise import generate_confusion_matrix, load_allfiles, build_attribute_level_metrics, build_category_level_metrics, build_weighted_avg_per_attribute, build_pipeline_overall_summary, get_master_accuracy, visualise_preprocessing_asheatmap, routingaccuracy


BASE_PATH = "D:/Dissertation/Final/Data/Segmented_Files/Final_Segmented_Files/results_tagging"
METHODS = ["Direct", "Descriptive"]
DIMMING = ["segmented_b", "full_dimmed", "seg_dimmed"]

def prepaer_paths():

    paths = [{
        "model": "SigLip",
        "Pipeline": f"Pipeline B",
        "Background Context": dim,
        "Prompt Method": method,
        "summary_file_path": f"{BASE_PATH}/Pipeline_B/SigLip/{method}/{dim}_summary_metrics.csv",
        "per_class_path": f"{BASE_PATH}/Pipeline_B/SigLip/{method}/{dim}_metrics_per_class.csv",
        "master_path":  f"{BASE_PATH}/Pipeline_B/SigLip/{method}/{dim}_master.csv"
    }
    for method in METHODS
    for dim in DIMMING
    ]
    return paths


if __name__ == "__main__":
    paths = prepaer_paths()
    summary_df, class_df, master_df = load_allfiles(paths)
    print(summary_df.head())
    attributes_to_drop = ["predicted_base_style", "predicted_colour"]
    summary_df = summary_df[~summary_df["attribute"].isin(attributes_to_drop)]
    master_df.to_csv(f"{BASE_PATH}/Fina.csv)")

    main_table = build_category_level_metrics(class_df)
    table_path = f"{BASE_PATH}/Final_Preprocessing_category_level_average.csv"
    main_table.to_csv(table_path)

    # main_table = build_attribute_level_metrics(summary_df)
    # table_path = f"{BASE_PATH}/'Final_attribute_level_average.csv'"
    # main_table.to_csv(table_path)

    main_table = build_weighted_avg_per_attribute(summary_df)
    table_path = f"{BASE_PATH}/Final_Preprocessing_weighted_attribute_level_average.csv"
    main_table.to_csv(table_path)
    visualise_preprocessing_asheatmap(BASE_PATH, "Final_Preprocessing_weighted_attribute_level_average.csv")

    routingaccuracy(master_df)

    main_table = build_pipeline_overall_summary(summary_df).T
    table_path = f"{BASE_PATH}/Final_Preprocessing_weighted_overall.csv"
    main_table.to_csv(table_path)

    main_table = get_master_accuracy(master_df)
    table_path = f"{BASE_PATH}/Final_Preprocessing_overall_accuracy.csv"
    main_table.to_csv(table_path)

    main_table = generate_confusion_matrix(BASE_PATH, "tags.garment_length_class", "predicted_length", master_df)
    