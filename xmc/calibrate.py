
def calibrate(model, optimizer, trainloader, validloader, testloader, trainloader_da, model2_path, args):
    best_p5 = 0
    num_stop_dropping = 0
    for epoch in range(0, args.epoch + 5):
        train_loss = model.one_epoch(epoch, trainloader, optimizer, mode='train', args=args)
        if epoch >= args.warmup and args.adjust:#calibration warm up
            if args.mda:
                model.one_epoch(epoch, trainloader_da, optimizer, mode='train', rebalance=True, args=args)
            else:
                model.one_epoch(epoch, trainloader_da, optimizer, mode='train', args=args)
        ev_result = model.one_epoch(epoch, validloader, optimizer, mode='eval', args=args)
        if args.test_each_epoch:
            test_result = model.one_epoch(epoch, testloader, optimizer, mode='eval')
            print(
                f'Epoch: {epoch} | Early Stop: {num_stop_dropping} | Train Loss: {train_loss: .4f} | Valid Result: {ev_result} | Test Result: {test_result}')
        else:
            print(
                f'Epoch: {epoch} | Early Stop: {num_stop_dropping} | Train Loss: {train_loss: .4f} | Valid Result: {ev_result}')
        p5 = ev_result[-1]
        if best_p5 < p5:
            best_p5 = p5
            model.save_model(model2_path)
            num_stop_dropping = 0
        else:
            num_stop_dropping += 1
        if num_stop_dropping >= args.early_stop_tolerance:
            print('Have not increased for %d check points, early stop training' % num_stop_dropping)
            break

