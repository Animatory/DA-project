import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm_notebook as tqdm


def train_cycle(num_epoch, loaders, model_name,
                student_model, teacher_model,
                student_optimizer, teacher_optimizer,
                criterion_sup, criterion_unsup,
                scheduler, metrics, summary_writer,  best_result=0, unsup_weight=3):
    for epoch in range(num_epoch):

        print(f'Start epoch {epoch}')
        for mode, loader in loaders.items():
            train_mode = mode == 'train'
            if train_mode:
                student_model.train()
                teacher_model.train()
            else:
                student_model.eval()
                teacher_model.eval()
            total_batches = len(loader)
            t = tqdm(enumerate(loader), total=total_batches)
            for i, batch in t:
                student_optimizer.zero_grad()

                with torch.set_grad_enabled(train_mode):
                    for key in batch:
                        batch[key] = batch[key].cuda()

                    if train_mode:
                        keys = ['source_image', 'target_image_0', 'target_image_1', 'source_label']
                        x_src, x_tgt0, x_tgt1, y = [batch[key] for key in keys]

                        logits_arcface, logits = student_model.forward_two_heads(x_src, y)
                        logits = torch.cat([logits_arcface, logits])
                        y = torch.cat([y, y])

                        student_logits_out = student_model(x_tgt0)
                        student_prob_out = F.softmax(student_logits_out, dim=1)
                        teacher_logits_out = teacher_model(x_tgt1)
                        teacher_prob_out = F.softmax(teacher_logits_out, dim=1)

                        unsup_loss = criterion_unsup(student_prob_out, teacher_prob_out)

                    else:
                        x_src, y = batch['input'], batch['target']
                        logits = teacher_model.forward(x_src)
                        unsup_loss = 0

                    clf_loss = criterion_sup(logits, y)

                    loss = clf_loss + unsup_loss * unsup_weight
                    loss_value = float(loss.data)

                    if train_mode:
                        loss.backward()
                        student_optimizer.step()
                        teacher_optimizer.step()
                        if isinstance(scheduler, lr_scheduler.CyclicLR):
                            scheduler.step()

                    metrics[-1].update(loss_value)
                    for metric in metrics[:-1]:
                        metric.update(logits, y)
                t.set_description_str(desc=f'Loss={metrics[-1].avg:.3f}', refresh=False)
            t.close()
            print(f'{mode} loss: {metrics[-1].avg:.3f}')
            for i, metric in enumerate(metrics):
                m_name = metric.name
                m_value = metric.value()
                if i != len(metrics) - 1:
                    print(f'{m_name}: {m_value}')
                summary_writer.add_scalar(f'{mode}/{m_name}', m_value, epoch + 1)

            if mode == 'target_valid':
                new_result = metrics[0].value()
                print("Previous best score: ", best_result, "Current score: ", new_result)
                if best_result <= new_result:
                    best_result = new_result
                    if new_result > 0.9:
                        torch.save(teacher_model.state_dict(),
                                   f'checkpoints/{model_name}_{int(round(best_result * 1000))}.pth')
                if not (scheduler is None or isinstance(scheduler, lr_scheduler.CyclicLR)):
                    scheduler.step()

            torch.save(teacher_model.state_dict(), f'checkpoints/{model_name}_last.pth')

            for metric in metrics:
                metric.reset()

        print()
