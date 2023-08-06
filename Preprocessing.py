import os
from re import finditer, search

import pandas as pd

from Model import tfidf_transformation, tfidf_kmeans_transformation, similarity_hierarchy_transformation, \
    similarity_k_means

module_label_dict = {
    "terraform-aws-modules/sns/aws": "sns",
    "terraform-aws-modules/ecs/aws": "ecs",
    "terraform-aws-modules/cloudwatch/aws//modules/log-group": "cloudwatch",
    "terraform-aws-modules/s3-bucket/aws": "s3",
    "terraform-aws-modules/lambda/aws": "lambda",
    "terraform-aws-modules/iam/aws//modules/iam-assumable-role": "iam_role",
    "terraform-aws-modules/iam/aws//modules/iam-policy": "iam-policy",
    "git::https://github.com/terraform-aws-modules/terraform-aws-s3-bucket?refv3.3.0": "s3",
    "git::https://github.com/terraform-aws-modules/terraform-aws-eventbridge?refv1.14.1": "eventbridge",
    "git::https://github.com/terraform-aws-modules/terraform-aws-lambda?refv3.3.1": "lambda",
    "git::https://github.com/terraform-aws-modules/terraform-aws-alb?refv7.0.0": "alb",
    "git::https://github.com/terraform-aws-modules/terraform-aws-ec2-instance?refv4.0.0": "ec2",
    "git::https://github.com/terraform-aws-modules/terraform-aws-route53//modules/records?refv2.8.0": "r53",
    "https://github.com/terraform-aws-modules/terraform-aws-rds?refv4.4.0": "rds",
    "git::https://github.com/terraform-aws-modules/terraform-aws-s3-bucket.git?refv2.10.0": "s3",
    "git::https://github.com/terraform-aws-modules/terraform-aws-ecr.git?refv1.1.1": "ecr",
    "git::https://github.com/terraform-aws-modules/terraform-aws-cloudwatch//modules/log-group?refv3.0.0": "cloudwatch",
    "git::https://github.com/terraform-aws-modules/terraform-aws-eventbridge.git?refv1.14.0": "eventbridge",
    "git::https://github.com/terraform-aws-modules/terraform-aws-s3-bucket.git?refv3.2.3": "s3",
    "git::https://github.com/terraform-aws-modules/terraform-aws-autoscaling?refv6.5.1": "autoscaling",
    "git::https://github.com/terraform-aws-modules/terraform-aws-iam//modules/iam-assumable-role?refv5.3.1": "iam_role",
    "git::https://github.com/terraform-aws-modules/terraform-aws-rds?refv4.4.0": "rds",
    "terraform-aws-modules/eks/aws": "eks",
    "terraform-aws-modules/s3-bucket/aws//modules/notification": "s3_notification",
    "terraform-aws-modules/sqs/aws": "sqs",
    "terraform-aws-modules/security-group/aws": "security-group",
    "terraform-aws-modules/vpc/aws": "vpc",
    "terraform-aws-modules/acm/aws": "acm",
    "../../": "unknown",
    "../..": "unknown",
    "terraform-aws-modules/cloudwatch/aws//examples/fixtures/aws_sns_topic": "sns_topic",
    "../../modules/notification": "s3_notification",
    "terraform-aws-modules/route53/aws//modules/records": "r53 records",
}


def version_source_removal(string_content: str):
    for string_loc in reversed(
            list(
                finditer(
                    r"source = |version = ",
                    string_content))):
        starting_location = string_loc.span()[0]
        char_length = 0
        quote_count = 0
        for char in string_content[string_loc.span()[1]:]:
            char_length += 1
            if char == '"':
                if quote_count == 0:
                    quote_count += 1
                else:
                    break
        string_content = (
                string_content[:starting_location]
                + string_content[string_loc.span()[1] + char_length:]
        )
    return string_content


def module_processing(payload, dir_content):
    title_loc = search(
        r"source[ ]+= \"[\w0-9./:=?-]+\"", payload)
    title = (
        payload[title_loc.span()[0] + 6: title_loc.span()[1] - 1]
        .replace(" ", "")
        .replace("=", "")
        .replace('"', "")
    )
    payload = version_source_removal(payload).strip()
    try:
        # print("Logs: Retrieving the label of module")
        label = module_label_dict[title]
        print("Logs: returning module " + label)
        return tuple((label, payload, dir_content))
    except KeyError:
        print("Error: Missing source label\n" + title)


def resource_data_processing(payload, dir_content):
    resource_name_loc = search('"[\\w]+"', payload)
    resource_name = payload[resource_name_loc.span(
    )[0] + 1: resource_name_loc.span()[1] - 1]
    return tuple((resource_name, payload, dir_content))


def tag_removal(string_content: str):
    for string_loc in reversed(
            list(
                finditer(
                    r"\w+tags = |tags = ",
                    string_content))):
        starting_location = string_loc.span()[0]
        bracket_stack = []
        char_length = 0
        for char in string_content[string_loc.span()[1]:]:
            char_length += 1
            if char not in ["m", "e", "r", "g", "e"]:
                if char == "{" or char == "[" or char == "(":
                    bracket_stack.append(char)
                elif char == "]" or char == "}" or char == ")":
                    bracket_stack.pop()
                if len(bracket_stack) == 0:
                    break
        string_content = (
                string_content[:starting_location]
                + string_content[string_loc.span()[1] + char_length:]
        )
    return string_content


def list_tuple_to_dataframe(list_of_tuples, resource_type: str):
    df = pd.DataFrame.from_records(list_of_tuples, columns=["name", "content", "file_loc"])
    df = df[df['name'].notnull()]
    unique_resource_name_list = df["name"].unique().tolist()
    if not os.path.exists(resource_type):
        os.makedirs(resource_type, exist_ok=True)
    for resource_name in unique_resource_name_list:
        print(resource_name)

        df_resource = df[df['name'] == resource_name]
        if df_resource.shape[0] > 1:
            save_dir = resource_type + "/" + resource_name + "/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            tfidf_transformation(df_resource)
            # similarity transformation
            similarity_hierarchy_transformation(df_resource)
            # tfidf similarity
            tfidf_kmeans_model = tfidf_kmeans_transformation(df_resource, save_dir)
            df_resource["tfidf_cluster"] = tfidf_kmeans_model.labels_.tolist()
            # similarity kmeans
            similarity_kmeans_model = similarity_k_means(df_resource, save_dir)
            df_resource["s_cluster"] = similarity_kmeans_model.labels_.tolist()

            df_resource.to_csv(save_dir + "result.csv")
    return df


def split_content(dirs_of_tf_files):
    module_list = []
    data_list = []
    resource_list = []
    for dir_file in dirs_of_tf_files:
        with open(dir_file) as file:
            payload_content = file.read()
            string_stop_loc = len(payload_content)
            payload_content = tag_removal(payload_content)
            for string_loc in reversed(
                    list(
                        finditer(
                            r"module \"|data \"|locals \{|resource \"",
                            payload_content,
                        )
                    )
            ):
                block_code = payload_content[string_loc.span()[0]: string_stop_loc]
                if block_code.rfind('module "') >= 0:
                    module_list.append(module_processing(block_code, dir_file))
                elif block_code.rfind('data "') >= 0:
                    data_list.append(resource_data_processing(block_code, dir_file))
                elif block_code.rfind('resource "') >= 0:
                    resource_list.append(resource_data_processing(block_code, dir_file))
                string_stop_loc = string_loc.span()[0]
    list_tuple_to_dataframe(module_list, "module")
    list_tuple_to_dataframe(data_list, "data")
    list_tuple_to_dataframe(resource_list, "resource")
