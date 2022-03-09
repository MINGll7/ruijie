
def evaluate(joint_model, data_loader, intent_metric, slot_metric):
    
    joint_model.eval()
    intent_metric.reset()
    slot_metric.reset()
    for idx, batch_data in enumerate(data_loader):
        input_ids, token_type_ids, intent_labels, tag_ids = batch_data
        intent_logits, slot_logits = joint_model(input_ids, token_type_ids=token_type_ids)
        # count intent metric
        intent_metric.update(pred_labels=intent_logits, real_labels=intent_labels)
        # count slot metric
        slot_pred_labels = slot_logits.argmax(axis=-1)
        slot_metric.update(pred_paths=slot_pred_labels, real_paths=tag_ids)

    intent_results = intent_metric.get_result()
    slot_results = slot_metric.get_result()

    return intent_results, slot_results