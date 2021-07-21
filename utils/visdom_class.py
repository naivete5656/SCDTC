import torch


class VisdomClass(object):
    def create_vis_show(self):
        return self.vis.images(
            torch.ones((self.batch_size, 1, 256, 256)), self.batch_size
        )

    def update_vis_show(self, images, window1):
        self.vis.images(images, self.batch_size, win=window1)

    def create_vis_plot(self, _xlabel, _ylabel, _title, _legend):
        return self.vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(xlabel=_xlabel, ylabel=_ylabel, title=_title, legend=_legend),
        )

    def update_vis_plot(self, iteration, loss, window1, update_type):
        self.vis.line(
            X=torch.ones((1)).cpu() * iteration,
            Y=torch.Tensor(loss).unsqueeze(0).cpu(),
            win=window1,
            update=update_type,
        )

    def vis_show_result(self, iteration, loss, mask_preds, imgs, true_masks, bg_mask):
        self.update_vis_plot(
            iteration, [loss.item()], self.iter_plot, "append"
        )
        mask_preds = (mask_preds - mask_preds.min()) / (
                mask_preds.max() - mask_preds.min()
        )
        self.update_vis_show(imgs.cpu(), self.ori_view)
        self.update_vis_show(mask_preds, self.pred_view)
        self.update_vis_show(true_masks.cpu(), self.gt_view)
        if self.pseudo:
            self.update_vis_show(bg_mask.cpu(), self.bg_mask_view)

    def vis_init(self):
        import visdom

        HOSTNAME = "localhost"
        PORT = 8097

        self.vis = visdom.Visdom(port=PORT, server=HOSTNAME, env=self.env)

        vis_title = "ctc"
        vis_legend = ["Loss"]
        vis_epoch_legend = ["Loss", "Val Loss"]

        self.iter_plot = self.create_vis_plot(
            "Iteration", "Loss", vis_title, vis_legend
        )
        self.epoch_plot = self.create_vis_plot(
            "Epoch", "Loss", vis_title, vis_epoch_legend
        )
        self.ori_view = self.create_vis_show()
        self.gt_view = self.create_vis_show()
        self.pred_view = self.create_vis_show()
        self.bg_mask_view = self.create_vis_show()
