import csv


def csv_evaluate_old(path: str, evaluation: dict):
    f = csv.writer(open(path, 'w'), delimiter=';')
    f.writerow(('class', 'dice', 'prd_pixes', 'lab_pixes', 'inter_pixes'))
    for i, cls in enumerate(['bg', 'stroma', 'normal', 'tumor', 'necrosis', 'vessel']):
        f.writerow((
            cls,
            '%.3f' % evaluation['dice_pixes'][i],
            '%d' % evaluation['pre_pixes'][i],
            '%d' % evaluation['lab_pixes'][i],
            '%d' % evaluation['inter_pixes'][i],
        ))


def csv_evaluate(path: str, evaluation: dict):
    f = csv.writer(open(path, 'w'), delimiter=';')
    f.writerow(('class', 'dice', 'prd_area', 'lab_area', 'inter_area'))

    f.writerow((
        'normal',
        '%.3f' % evaluation['normal']['area']['dice'],
        '%d' % evaluation['normal']['area']['pre'],
        '%d' % evaluation['normal']['area']['lab'],
        '%d' % evaluation['normal']['area']['inter'],
    ))
    f.writerow((
        'atrophic',
        '%.3f' % evaluation['atrophic']['area']['dice'],
        '%d' % evaluation['atrophic']['area']['pre'],
        '%d' % evaluation['atrophic']['area']['lab'],
        '%d' % evaluation['atrophic']['area']['inter'],
    ))
    f.writerow(('-',) * 5)

    f.writerow(('class', 'iou_th', 'TP', 'FP', 'FN', 'PRE', 'REC', 'F.5', 'F1', 'F2'))
    for cls in ['normal', 'atrophic']:
        for th in ['0.3', '0.5', '0.8']:
            f.writerow((
                cls,
                th,
                evaluation[cls][th]['TP'],
                evaluation[cls][th]['FP'],
                evaluation[cls][th]['FN'],
                evaluation[cls][th]['pre'],
                evaluation[cls][th]['rec'],
                evaluation[cls][th]['f05'],
                evaluation[cls][th]['f1'],
                evaluation[cls][th]['f2'],
            ))
        f.writerow(('-',) * 10)
