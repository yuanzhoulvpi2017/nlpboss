import pandas as pd
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers import TrainingArguments
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from typing import List, Tuple


class SendEmailCallback(TrainerCallback):
    """
    :parameter
        password: <email password>
        account: <email>
        receive_email: [<receive_email>, <reveive_email>>]

    ```python
    send_key = "fmlxxxxxxxxfjh"
    send_email = "15820xxxx@qq.com"
    receive_email = ["yuaxxxxxx@outlook.com", "huxxxxxx@icloud.com", "158xxxxx@qq.com"]

    mlc = SendEmailCallback(password=send_key, account=send_email, receive_email=receive_email)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_text["train"],
        eval_dataset=tokenized_text["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[mlc]
    )

    trainer.train()
    ```

    """

    def __init__(self, password: str, account: str, receive_email: List[str]) -> None:
        super().__init__()
        self.password = password
        self.account = account
        self.receive_email = receive_email

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):
        """
        Event called after logging the last logs.
        """
        logdata = state.log_history
        self.send(log=logdata)

    def split_data(self, log) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        data = pd.DataFrame(log)

        data_part1 = data[['loss', 'learning_rate', 'epoch', 'step']].dropna(subset=['learning_rate'])

        try:
            data_part2 = data.pipe(
                lambda x: x[['epoch', 'step'] + [i for i in x.columns.tolist() if i.startswith('eval')]]
            ).dropna(subset=['eval_runtime'])

            if data_part2.shape[1] > 2 and data_part2.shape[0] > 0:
                data_part2 = data_part2
            else:
                data_part2 = pd.DataFrame()

        except Exception as e:
            data_part2 = pd.DataFrame()

        try:

            data_part3 = data.pipe(
                lambda x: x[['epoch', 'step'] + [i for i in x.columns.tolist() if i.startswith('train')]]
            ).dropna(subset=['train_runtime'])

            if data_part3.shape[1] > 2 and data_part3.shape[0] > 0:
                data_part3 = data_part3
            else:
                data_part3 = pd.DataFrame()

        except Exception as e:
            data_part3 = pd.DataFrame()

        return data_part1, data_part2, data_part3

    def send(self, log):
        allmessage = MIMEMultipart("alternative")
        allmessage['From'] = Header("MOSS", 'utf-8')
        allmessage['To'] = Header("ME", 'utf-8')
        allmessage['Subject'] = Header("Train Info", 'utf-8')

        data1, data2, data3 = self.split_data(log=log)

        for (_, data) in enumerate([data1, data2, data3]):

            if data.shape[0] > 0:
                body_email_text = data.to_html()
                body_email = MIMEText(body_email_text, "html", "utf-8")
                allmessage.attach(body_email)

        server = smtplib.SMTP_SSL("smtp.qq.com", 465)

        server.login(self.account, self.password)

        server.sendmail(self.account, self.receive_email, allmessage.as_string())
        server.quit()
