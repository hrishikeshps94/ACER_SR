import torch
from collections import OrderedDict
import numpy as np
def parameter_count(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_non_trainable_params = pytorch_total_params - pytorch_total_trainable_params
    print(f'Total parameters = {pytorch_total_params}\n Tainable parameters = {pytorch_total_trainable_params}\n Non Trainable parametrs = {pytorch_non_trainable_params}\n')
    return None

def get_model_summary(
    model, input_size, batch_size=-1, device=torch.device("cpu"), dtypes=None
):
    """Reusable utility layers such as pool or upsample will also get printed, but their printed values will
    be corresponding to the last call.

    """
    if dtypes == None:
        dtypes = [torch.FloatTensor] * len(input_size)

    summary_str = ""

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = module.name if module.name != "" else "%s" % class_name

            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            elif isinstance(output, dict):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output.values()
                ]
            elif isinstance(output, torch.Tensor):
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [
        torch.rand(2, *in_size).type(dtype).to(device=device)
        for in_size, dtype in zip(input_size, dtypes)
    ]

    # create properties
    summary = OrderedDict()
    hooks = []

    # create layer name according to hierachy names
    for name, module in model.named_modules():
        module.name = name

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # aligning name to the left
    max_name_length = len(max(summary.keys(), key=len))
    summary = [(k.ljust(max_name_length), v) for k, v in summary.items()]
    summary = OrderedDict(summary)

    # remove these hooks
    for h in hooks:
        h.remove()

    header_line = "{}  {:>25} {:>15}".format(
        "Layer Name".center(max_name_length), "Output Shape", "Param #"
    )
    summary_str += "".join("-" for _ in range(len(header_line))) + "\n"
    summary_str += header_line + "\n"
    summary_str += "".join("=" for _ in range(len(header_line))) + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(
        np.prod(sum(input_size, ())) * batch_size * 4.0 / (1024 ** 2.0)
    )
    total_output_size = abs(
        2.0 * total_output * 4.0 / (1024 ** 2.0)
    )  # x2 for gradients
    total_params_size = abs(total_params * 4.0 / (1024 ** 2.0))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "".join("=" for _ in range(len(header_line))) + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += (
        "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    )
    summary_str += "".join("-" for _ in range(len(header_line))) + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "".join("-" for _ in range(len(header_line))) + "\n"
    return summary_str