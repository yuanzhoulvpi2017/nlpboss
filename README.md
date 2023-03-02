




## 操作

### 使用QQ邮箱📮

这里提供一个QQ邮箱的生成`生成授权码`的教程：[https://zhuanlan.zhihu.com/p/356769096](https://zhuanlan.zhihu.com/p/356769096)

### 初始化`callback`
```python 

from nlpboss.callback import SendEmailCallback

send_key = "fmlkaxxxxxxxjfjh" # 这个是账号的密码
send_email = "1582034172@qq.com" # 这个是账号

receive_email = ["yuanxxxxxx@outlook.com", "huxxxxxxx@icloud.com", "1582034172@qq.com"] # 这个是需要发送给的人

mlc = SendEmailCallback(password=send_key, account=send_email, receive_email=receive_email)

```

### 把`callback`放到`Trainer`的里面
```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_text["train"],
    eval_dataset=tokenized_text["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[mlc]  #<------------------------------------ 在这里写上实例化的对象
)

trainer.train()



```