//
// Created by patrick on 4/4/19.
//


namespace lvr2{


    //TODO: use REDBLACKTREE

    /**
     * Deletes the (subtree) beginning with cell c
     * @param c
     * @return
     */
    Cell* TumbleTree::makeEmpty(Cell* c)
    {
        if(c == NULL)
            return NULL;
        {
            makeEmpty(c->left);
            makeEmpty(c->right);
            delete c;
        }
        return NULL;
    }

    Cell* TumbleTree::insertIterative(float sc, VertexHandle vH)
    {

        Cell* newCell = new Cell;
        newCell->signal_counter = sc;
        newCell->duplicateMap.insert(vH,sc);
        newCell->left = NULL;
        newCell->right = NULL;

        if(root == NULL)
        {
            root = newCell;
            return root;
        }

        Cell* tmp = root;

        Cell* maintain = NULL;

        while(sc != tmp->signal_counter)
        {
            if(sc < tmp->signal_counter)
            {
                if(tmp->left == NULL)
                {
                    tmp->left = newCell;

                    return tmp->left;
                }
                tmp = tmp->left;
            }
            else if(sc > tmp->signal_counter)
            {
                if(tmp->right == NULL)
                {
                    tmp->right = newCell;

                    return tmp->right;
                }
                tmp = tmp->right;
            }
        }

        //found
        tmp->duplicateMap.insert(vH,sc);

        return tmp;

        /*while(tmp != NULL) //while the current cell is not the one we are looking for
        {
            maintain = tmp;
            if(sc < tmp->signal_counter){ //dive into the left subtree, if its non-NULL, else insert a new cell
                tmp = tmp->left;
            }
            else if(sc > tmp->signal_counter){ //dive into the right subtree, if its non-NULL, else insert a new cell
                tmp = tmp->right;
            }
            else if(sc == tmp->signal_counter){ //if the sc is found, we insert into the map.
                maintain->duplicateMap.insert(vH,sc);
                std::cout << "abcdefg" <<  maintain->duplicateMap.numValues() << endl;
                return maintain;
            }
        }

        //if the root is null
        if(maintain == NULL)
        {
            maintain = newCell;
        }
        else if(sc < maintain->signal_counter)
        {
            maintain->left = newCell;
            return maintain->left;
        }
        else
        {
            maintain->right = newCell;
            return maintain->right;
        }

        return maintain;*/
    }

    //TODO: maybe an iterative method would be better, as it really returns the cell we want to map
    Cell* TumbleTree::insert(float sc, VertexHandle vH, Cell* c)
    {

        if(c == NULL)
        {
            c = new Cell;
            c->signal_counter = sc;
            c->duplicateMap.insert(vH, sc);
            c->left = c->right = NULL;
        }
        else if(sc < c->signal_counter)
        {
            c->left = insert(sc, vH, c->left);
        }
        else if(sc > c->signal_counter)
        {
            c->right = insert(sc, vH, c->right);
        }
        else
        {
            c->duplicateMap.insert(vH,sc);
        }

        return c; //return the inserted Cell
    }

    void TumbleTree::removeIterative(float sc, VertexHandle vH, bool removeWhole)
    {

        if(root == NULL){
            return;
        }

        //first: find cell
        //remove cell and set min of right subtree as the new subroot, if only one vertex remains in the duplicate map
        Cell* tmp = root;
        Cell* parent = NULL;
        bool direction = false; //false : left, true: right
        while(tmp->signal_counter != sc)
        {
            if(sc < tmp->signal_counter)
            {
                parent = tmp;
                direction = false;
                tmp = tmp->left;
            }
            else if(sc > tmp->signal_counter)
            {
                parent = tmp;
                direction = true;
                tmp = tmp->right;
            }
            if(tmp == NULL) return; //not found
        }
        //now we are at the correct cell.

        if(tmp->left && tmp->right) //if we are somewhere in the middle of the tree
        {
            tmp->duplicateMap.erase(vH);
            if(tmp->duplicateMap.numValues() == 0)
            {
                //if there are no values left in the duplicate map, the cell needs to be removed
                Cell* minRightSubtree = this->findMin(tmp->right);
                tmp->signal_counter = minRightSubtree->signal_counter;
                //update duplicate map (copy values from other map)
                for(auto iter = minRightSubtree->duplicateMap.begin(); iter != minRightSubtree->duplicateMap.end();++iter)
                {
                    tmp->duplicateMap.insert(*iter,minRightSubtree->signal_counter);
                }
                VertexHandle ret(0);
                cout << "Min found, trying to remove it..." << minRightSubtree->signal_counter << endl;
                removeIterative(minRightSubtree->signal_counter, ret,true);
            }
        }
        else{
            if(!removeWhole) tmp->duplicateMap.erase(vH); //no need to erase, if the whole cell should be erased
            if(tmp->duplicateMap.numValues() == 0 || removeWhole)
            {
                std::cout << "One subtree..." << endl;
                Cell* toBeDeleted = tmp;
                if(tmp->left){
                    tmp = tmp->left;
                }
                else if(tmp->right)
                {
                    tmp = tmp->right;
                }
                if(parent){
                    cout << "parenting" << endl;
                    direction ? parent->right = tmp : parent->left = tmp;
                }
                delete toBeDeleted;

            }
        }
    }

    //TODO: make it work. expected number of cells after the algorithm: runtime*numsplits iterative??
    Cell* TumbleTree::remove(float sc, VertexHandle vH, Cell* c, bool removeWhole)
    {
        Cell* tmp;
        if(c == NULL)
        {
            c != root ? notDeleted++ : notDeleted = notDeleted-1+1;
            return NULL;
        }
        else if(sc < c->signal_counter)
        {
            c->left = remove(sc, vH, c->left, removeWhole);
        }
        else if(sc > c->signal_counter) {
            c->right = remove(sc, vH, c->right, removeWhole);
        }
        else{
            //if there are two are more, just remove from the duplicate map
            if(c->duplicateMap.numValues() > 1 && !removeWhole){
                int numV = c->duplicateMap.numValues();
                c->duplicateMap.erase(vH);
                if(c->duplicateMap.numValues() == numV){
                    notDeleted++;
                }
                return c;
            }

            //if there is one or no child
            if(c->left == NULL)
            {
                tmp = c->right;
                delete c;
                return tmp;
            }
            else if(c->right == NULL)
            {
                tmp = c->left;
                delete c;
                return tmp;
            }
            //node with two children: get min of the right subtree (inorder successor of right subtree)
            tmp = findMin(c->right);

            //copy data from inorder successor
            c->signal_counter = tmp->signal_counter;
            c->duplicateMap.clear();
            //copy values from one dupilcate map to the other
            for(auto iter = tmp->duplicateMap.begin(); iter != tmp->duplicateMap.end();++iter)
            {
                c->duplicateMap.insert(*iter,tmp->signal_counter);
            }
            //remove inorder successor
            VertexHandle ret(0);
            c->right = remove(tmp->signal_counter, ret, c->right, true);

        }

        return c;
    }

    /**
     * finds the cell with the minimum sc
     * @param c starting cell
     * @return cell with min sc
     */
    Cell* TumbleTree::findMin(Cell* c)
    {
        if(c == NULL)
            return NULL;
        else if(c->left == NULL)
            if(c != root)
                return c;               // TODO: update it after the sc update is O(log(n))
            else
                return findMin(c->right);
        else
            return findMin(c->left);
    }

    /**
     * finds the cell with the maximum sc
     * @param c starting cell
     * @return cell with max sc
     */
    Cell* TumbleTree::findMax(Cell* c, int* depth)
    {
        if(c == NULL)
            return NULL;
        else if(c->right == NULL)
            if(c != root){
                //std::cout << "Depth finding max: " << *depth << endl;
                return c;
            }
            else{
                *depth = *depth + 1;
                return findMax(c->left, depth);
            }
        else{
            *depth = *depth + 1;
            return findMax(c->right, depth);
        }

    }

    Cell* TumbleTree::find(float sc, VertexHandle vH, Cell* c)
    {
        if(c == NULL)
        {
            return NULL;
        }
        else if(sc < c->signal_counter)
        {
            return find(sc, vH, c->left);
        }
        else if(sc > c->signal_counter)
        {
            return find(sc, vH, c->right);
        }
        else
        {
            if(c->duplicateMap.containsKey(vH))
                return c;
            else
            {
                cout << "Cell found, though it doesnt contain the found handle: " << vH.idx() << endl;
                cout << "It contains: " << endl;
                for(auto iter = c->duplicateMap.begin(); iter != c->duplicateMap.end(); ++iter)
                {
                    cout << *iter << " ";
                }
                cout << endl;
                return NULL; //if the key does not exist in the cell with the suitable signal counter
            }
        }

    }

    void TumbleTree::inorder(Cell* c)
    {
        if(c == NULL)
            return;
        if(c->duplicateMap.numValues() == 0) cout <<"dafuq"<<endl;
        inorder(c->left);
        std::cout << " | [";
        for(auto iter = c->duplicateMap.begin(); iter!= c->duplicateMap.end(); ++iter)
        {
            cout << c->duplicateMap[*iter] << ", ";
        }
        std::cout << "]";
        inorder(c->right);
    }

    //update the SCs (righ now linear -> O(n), later O(log(n))
    void TumbleTree::update(Cell *c, float alpha, VertexHandle vH)
    {
        if(c == NULL) return;
        c->signal_counter -= c->signal_counter*alpha;
        for(auto iter = c->duplicateMap.begin(); iter != c->duplicateMap.end(); ++iter)
        {
            float& sc = c->duplicateMap.get(*iter).get();
            sc -= sc * alpha;
        }

        update(c->left, alpha, vH);
        update(c->right, alpha, vH);
    }

    int TumbleTree::size(Cell* c)
    {
        if(c == NULL) return 0;
        return (int)c->duplicateMap.numValues() + size(c->left) + size(c->right);
    }

    TumbleTree::TumbleTree()
    {
        root = NULL;
    }

    TumbleTree::~TumbleTree()
    {
        root = makeEmpty(root);

    }

    // public       ||
    // calls        \/

    void TumbleTree::insert(float sc, VertexHandle vH)
    {
        root = insert(sc, vH, this->root);
    }

    // we only need functionality to remove a specific cell. TODO: look, wheather or not it really returns  the root
    void TumbleTree::remove(Cell* c, VertexHandle vH)
    {
        root = remove(c->signal_counter, vH, root);
    }

    void TumbleTree::remove(float sc, VertexHandle vH)
    {
        root = remove(sc, vH, root);
    }

    void TumbleTree::display()
    {
        inorder(root);
        cout << endl;
    }


    Cell* TumbleTree::min()
    {
        return this->findMin(root);
    }

    Cell* TumbleTree::max()
    {
        return this->findMax(root);
    }

    Cell* TumbleTree::find(float sc, VertexHandle vH)
    {
        return find(sc, vH, root);
    }

    void TumbleTree::updateSC(float alpha, VertexHandle vH) {
        this->update(root, alpha, vH);
    }

    int TumbleTree::size(){
        if(root == NULL) return 0;
        return size(root);
    }

} // namespace lvr2