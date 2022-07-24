import torch

def test(options, model, vid_data_loader, text_data_loader, evaluation, metrics, logging, output_dir, norm_score, cal_perf, pred_error_matrix_file):
    # mapping
    video_embs, video_ids = evaluation.encode_text_or_vid(model.embed_vis, vid_data_loader)
    cap_embs, cap_trans_embs, caption_ids = evaluation.encode_text_hybrid(model.embed_txt, text_data_loader)

    v2t_gt, t2v_gt = metrics.get_gt(video_ids, caption_ids)

    logging.info("write into: %s" % output_dir)

    t2v_all_errors_1 = evaluation.cal_error(video_embs, cap_embs, options.measure)

    t2v_all_errors_2 = evaluation.cal_error(video_embs, cap_trans_embs, options.measure)

    for w in [1, 0.8, 0.5, 0.2, 0.0]:
        print(w, '------')
        t2v_all_errors_1 = norm_score(t2v_all_errors_1)
        t2v_all_errors_2 = norm_score(t2v_all_errors_2)
        t2v_tag_all_errors = w * t2v_all_errors_1 + (1 - w) * t2v_all_errors_2
        cal_perf(t2v_tag_all_errors, v2t_gt, t2v_gt)

    torch.save({'errors': t2v_tag_all_errors, 'videos': video_ids, 'captions': caption_ids}, pred_error_matrix_file)
    logging.info("write into: %s" % pred_error_matrix_file)