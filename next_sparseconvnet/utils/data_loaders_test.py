from . data_loaders import *

def test_DataGen_classification(labels_df):
    binsX = binsY = np.linspace(-200, 200, 11)
    binsZ = np.linspace(0, 550, 11)
    datagen = DataGen_classification(labels_df, binsX, binsY, binsZ)
    data = datagen[0]

    assert len(data) == 6 #x, y, z, ener, label, event

    assert data[0].dtype == np.int64
    assert data[1].dtype == np.int64
    assert data[2].dtype == np.int64
    assert data[3].dtype == np.float
    assert len(data[0])==len(data[1])==len(data[2])==len(data[3])

    assert data[4] in [0, 1]
    assert isinstance(data[5], np.int64)


def test_collatefn(labels_df):
    binsX = binsY = np.linspace(-200, 200, 11)
    binsZ = np.linspace(0, 550, 11)
    datagen = DataGen_classification(labels_df, binsX, binsY, binsZ)
    data = [datagen[i] for i in range(3)]
    batch = collatefn(data)

    assert len(batch) == 4 #coords, energies, labels, events
    coords = batch[0]
    energs = batch[1]
    assert coords.shape[1] == 4 #x, y, z, bid
    assert coords.dtype == torch.long
    assert coords.shape[0] == energs.shape[0]
    assert energs.dtype == torch.float