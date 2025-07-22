from torchprime.models import resnet18, transact

registry = {
  "pinterest/transformer_user_action": transact.get_model,
  "resnet18": resnet18.get_model,
}
