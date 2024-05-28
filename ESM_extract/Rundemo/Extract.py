#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        nargs="?",
        default= "esm1b_t33_650M_UR50S",
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )

    # parser.add_argument(
    #     "model_location",
    #     type=str,
    #     nargs="?",
    #     default="G:/Edge_DownLoad/esm1b_t33_650M_UR50S.pt",
    #     help="PyTorch model file OR name of pretrained model to download (see README for models)",
    # )

    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        nargs="?",
        default="G:/fkfk/AlphaFold_refine/ESM_extract/Rundemo/fasta_file/Q7L5Y9.fasta.txt", #G:\fkfk\AlphaFold_refine\ESM_extract\Rundemo\fasta_file
        help="FASTA file on which to extract representations",

    )#nargs="?" 表示参数是可选的，如果用户未提供则使用默认值。  nargs="+" 表示该参数应该接受一个或多个命令行参数，并将它们存储为一个列表。
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        nargs="?",
        default="G://fkfk//AlphaFold_refine//ESM_extract//Rundemo//output_dir",
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        #choices=["mean", "per_tok", "bos", "contacts"],
        default=["mean"],
        help="specify which representations to return",
        #required=True, #我已经用default提供了一个默认值，所以它不再是必须的
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate sequences longer than 1024 to match the training setup",
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def main(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include #true

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers) #确保 args.repr_layers 中的值符合一定的约束条件，下面一行代码对这些值进行了转换。
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    """
    with torch.no_grad():
    用于在 PyTorch 中禁止梯度计算。在这个上下文中，所有在其内部的操作都不会被记录梯度，即使在计算图构建的情况下，也不会跟踪操作的梯度信息。
    通常情况下，当你在进行推理或者评估阶段时，不需要计算梯度，因为这会消耗额外的计算资源并增加内存开销。此时，使用 torch.no_grad() 上下文管理器可以提高代码的执行效率和减少内存使用。
    """
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            if args.truncate:
                toks = toks[:, :1022]

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1 : len(strs[i]) + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in args.include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : len(strs[i]) + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone()

                torch.save(
                    result,
                    args.output_file,
                )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(args)
    main(args)