//
// Created by patrick on 4/4/19.
//


namespace lvr2{

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

    //TODO: make it work. expected number of cells after the algorithm: runtime*numsplits
    Cell* TumbleTree::remove(float sc, VertexHandle vH, Cell* c)
    {
        Cell* tmp;
        if(c == NULL)
        {
            return NULL;
        }
        else if(sc <= c->signal_counter)
        {
            c->left = remove(sc, vH, c->left);
        }
        else if(sc > c->signal_counter) {
            c->right = remove(sc, vH, c->right);
        }
        else if(c->left && c->right)
        {
            c->duplicateMap.erase(vH); //erase key and value in hashlist
            if(c->duplicateMap.numValues() == 0)
            {
                tmp = findMin(c->right); //find minimum sc cell of the right subtree (should be bigger than all left cells sc's)
                c->signal_counter = tmp->signal_counter; //copy sc and duplicate map
                c->duplicateMap = tmp->duplicateMap;
                c->right = remove(c->signal_counter, c->duplicateMap.begin().operator*(), c->right); //delete the minimum of the right subtree, as it now serves as the new subtree-root
            }

        }
        else{
            tmp = c;
            //now we only got one or no subtree left, if there
            if(c->duplicateMap.numValues() == 0)
            {
                if(c->left == NULL)
                {
                    c = c->right;
                }
                else if(c->right == NULL)
                {
                    c = c->left;
                }
                delete tmp;
            } else{
                c->duplicateMap.erase(vH);
            }

        }
        //return c;
    }

    /**
     * finds the cell with the minimum sc, might have a left subtree containing more of it, with equal sc's.
     * @param c starting cell
     * @return cell with min sc
     */
    Cell* TumbleTree::findMin(Cell* c)
    {
        if(c == NULL)
            return NULL;
        else if(c->left == NULL)
            return c;
        else
            return findMin(c->left);
    }

    /**
     * finds the cell with the maximum sc, might have a left subtree containing more of it, with equal sc's.
     * @param c starting cell
     * @return cell with max sc
     */
    Cell* TumbleTree::findMax(Cell* c)
    {
        if(c == NULL)
            return NULL;
        else if(c->right == NULL)
            return c;
        else
            return findMax(c->right);
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
        else if(c->duplicateMap.containsKey(vH))
        {
            return c;
        }
        else return NULL; //if the key does not exist in the cell with the suitable signal counter
    }

    void TumbleTree::inorder(Cell* c)
    {
        if(c == NULL)
            return;
        inorder(c->left);
        cout << c->signal_counter << " ";
        inorder(c->right);
    }

    //update the SCs (righ now linear -> O(n), later O(log(n))
    void TumbleTree::update(Cell *c, float alpha, VertexHandle vH)
    {
        if(c == NULL) return;
        if(!c->duplicateMap.containsKey(vH))
        {
            c->signal_counter -= c->signal_counter*alpha;
            for(auto iter = c->duplicateMap.begin(); iter!= c->duplicateMap.end(); ++iter)
            {
                float& sc = c->duplicateMap[*iter];
                sc -= sc * alpha;
            }
        }

        update(c->left, alpha, vH);
        update(c->right, alpha, vH);
    }

    int TumbleTree::size(Cell* c)
    {
        int i = (int)c->duplicateMap.numValues();
        if(c->right)
        {
            i += size(c->right);
        }
        if(c->left)
        {
            i += size(c->left);
        }

        return i;
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

    // we only need functionality to remove a specific cell.
    void TumbleTree::remove(Cell* c, VertexHandle vH)
    {
        root = remove(c->signal_counter, vH, this->root);
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
        return size(root);
    }

} // namespace lvr2