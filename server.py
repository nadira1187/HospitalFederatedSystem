import flwr as fl


def weighted_average(metrics):
    accuracies = [
        num_examples * m["accuracy"]
        for num_examples, m in metrics
    ]

    examples = [
        num_examples
        for num_examples, _ in metrics
    ]

    return {
        "accuracy": sum(accuracies) / sum(examples)
    }


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,

    min_fit_clients=4,
    min_evaluate_clients=4,
    min_available_clients=4,

    evaluate_metrics_aggregation_fn=weighted_average,
)


fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)