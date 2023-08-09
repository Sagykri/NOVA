notebook_name=$1

jupyter nbconvert --to notebook --inplace --execute $notebook_name.ipynb --ExecutePreprocessor.kernel_name=cytoself --allow-errors 
jupyter nbconvert --to html $notebook_name.ipynb