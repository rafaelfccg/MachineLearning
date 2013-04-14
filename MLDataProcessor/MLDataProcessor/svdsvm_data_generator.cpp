

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;


typedef vector<float> fvec;

int main()
{
	string label_file = "F:\\Projects\\GitHub\\galaxy-zoo\\dev\\database_labels\\cleaned_labels\\task1_super-clean.txt";
	string feat_dir = "I:\\classemes\\";

	ifstream label_in(label_file);
	string title;
	int sample_num, label_num;
	label_in>>title>>title>>title;
	label_in>>title>>sample_num>>label_num;

	// save feature vector for certain type of samples
	vector<vector<fvec>> samples(label_num);
	string asset_id;
	int label;
	for(int i=0; i<sample_num; i++)
	{
		cout<<i<<endl;

		label_in>>asset_id>>label;
		// read classemes for each file
		// rows, cols (1), column vector
		string feat_file = feat_dir + asset_id + "_classemes.dat";
		ifstream in(feat_file.c_str(), ios::binary);
		if(!in.good())
		{
			cout<<"No classemes file: "<<asset_id<<endl;
			getchar();
			return -1;
		}
		int rows, cols;
		in.read((char*)&rows, 4);
		in.read((char*)&cols, 4);
	
		// row order
		float *data = new float[rows*cols];
		in.read((char*)data, rows*cols*sizeof(float));

		//cout<<data[0]<<" "<<data[1]<<endl;
		// copy to vector structure
		fvec cur_feat(rows*cols, 0);
		for(int j=0; j<rows*cols; j++)
			cur_feat[j] = data[j];
		samples[label].push_back( cur_feat );

		delete data;
		data = NULL;
	}

	cout<<"Input all samples"<<endl;

	// select 80% samples from both sides as training file and left ones are test
	ofstream train_out("classemes1_train.txt");
	ofstream test_out("classemes1_test.txt");
	// multi-class case; better switch to binary label (-1/+1) for binary task
	for(int id=0; id<samples.size(); id++)
	{
		// loop all samples
		for(int i=0; i<samples[id].size(); i++)
		{
			if(i<samples[id].size()*0.8)
			{
				train_out<<id<<" ";
				for(int j=0; j<samples[id][i].size(); j++)
				{
					train_out<<(j==0? "": " ")<<j<<":"<<samples[0][i][j];
				}
				train_out<<endl;
			}
			else
			{
				test_out<<id<<" ";
				for(int j=0; j<samples[id][i].size(); j++)
				{
					test_out<<(j==0? "": " ")<<j<<":"<<samples[id][i][j];
				}
				test_out<<endl;
			}
		}
	}
	

	return 0;
}